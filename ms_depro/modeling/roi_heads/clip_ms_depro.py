import logging
import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import detectron2.utils.comm as comm

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, language_encoder):
        super().__init__()
        self.transformer = language_encoder.transformer
        self.positional_embedding = language_encoder.positional_embedding
        self.ln_final = language_encoder.ln_final
        self.text_projection = language_encoder.text_projection
        self.dtype = language_encoder.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MSPromptLearner(nn.Module):
    def __init__(self, 
                 classnames, 
                 language_encoder, 
                 csc, 
                 agnosticnet:bool = False, 
                 specificnet:bool = False, 
                 specificnet_from_bbox:bool = False, 
                 specificnet_test_with_buffer: bool = False,
                 metanet_shallow_features: bool = False, 
                 learnable_bg: bool = False
                 ):
        super().__init__()
        
        logger = logging.getLogger(__name__)
    
        self.csc = csc # class-specific context token.
        
        n_cls = len(classnames)
        n_ctx = 16
        
        n = n_ctx # TODO: hacky-way
        
        dtype = language_encoder.dtype
        ctx_dim = language_encoder.ln_final.weight.shape[0]

        if self.csc:
            if comm.is_main_process():
                print(f"[{__name__}] initialize class-specific context")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            if comm.is_main_process():
                print(f"[{__name__}] initialize a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n)

        if comm.is_main_process():
            print(f"[{__name__}] ctx vector size: {ctx_vectors.size()}")
            print(f"[{__name__}] initial context: {prompt_prefix}")
            print(f"[{__name__}] number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompts = [
            prompt_prefix + " " + name + "."
            for name in classnames
        ] 
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = language_encoder.token_embedding(tokenized_prompts).type(
                dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])   # SOS
        self.register_buffer("token_suffix", embedding[:,
                                                       1 + n:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts
          
        self.agnosticnet = agnosticnet
        if agnosticnet:
            from collections import OrderedDict
            vis_dim = 256 if metanet_shallow_features else 1024
            self.agnostic_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
            ]))
            self.n_agnostic = 8
            self.n_specific = n - self.n_agnostic
            
            self.register_buffer("agnostic_buffer", torch.zeros(1, 1, ctx_dim))
            self.ema_keep_rate = 0.99
            
        self.specificnet = specificnet
        self.specificnet_from_bbox = specificnet_from_bbox
        self.specificnet_test_with_buffer = specificnet_test_with_buffer

        if specificnet:
            from collections import OrderedDict
            vis_dim = 256 if metanet_shallow_features else 1024
            self.specific_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
            ]))
            self.n_agnostic = 8
            self.n_specific = n - self.n_agnostic
            
            self.register_buffer("specific_bbox_buffer", torch.zeros(1, 1, ctx_dim))
            self.specific_ema_keep_rate = 0.99
          
        
        self.learnable_bg = learnable_bg
        if learnable_bg:
            bg_prompt = torch.empty(n, ctx_dim, dtype=dtype)
            nn.init.normal_(bg_prompt, std=0.02)
            self.bg_prompt = nn.Parameter(bg_prompt)
            tokenized_bg_prompt = ' '.join(['X'] * (n))
            tokenized_bg_prompt = clip.tokenize(tokenized_bg_prompt)
            self.tokenized_prompts = torch.cat([self.tokenized_prompts, tokenized_bg_prompt])
            
    def update_agnostic_buffer(self, bias_agnostic, ema_keep_rate=0.99):
        if self.agnostic_buffer.sum()==0:
            self.agnostic_buffer = bias_agnostic
        else:
            self.agnostic_buffer = (ema_keep_rate*self.agnostic_buffer) + (1-ema_keep_rate)*bias_agnostic
    
    def update_specific_bbox_buffer(self, bias_specific, specific_ema_keep_rate=0.99):
        if self.specific_bbox_buffer.sum()==0:
            self.specific_bbox_buffer = bias_specific
        else:
            self.specific_bbox_buffer = (specific_ema_keep_rate*self.specific_bbox_buffer) + (1-specific_ema_keep_rate)*bias_specific
            
    def forward(self, domain_label, is_inference=False, agnostic_feats=None, specific_feats=None):
        ctx = self.ctx
        if ctx.dim() == 2:
            if not self.csc:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) 
                
        if self.learnable_bg:
            ctx = torch.cat([ctx, self.bg_prompt.unsqueeze(0)], dim=0)
        
        prefix = self.token_prefix 
        suffix = self.token_suffix
        
        if self.agnosticnet:
            if is_inference:
                bias_agnostic = self.agnostic_buffer
            else: 
                bias_agnostic = self.agnostic_net(agnostic_feats).mean(dim=0, keepdim=True).unsqueeze(1)  # (1, 1, ctx_dim)
                self.update_agnostic_buffer(bias_agnostic, self.ema_keep_rate)
            
            ctx_agnostic = ctx[:, :self.n_agnostic, :] + bias_agnostic
            ctx = torch.cat([ctx_agnostic, ctx[:, self.n_agnostic:, :]], dim=1)
            
        
        if self.specificnet:
            is_training = not is_inference

            # (i) branch=='supervised_source', (ii) branch=='pseudo_training_target', (iii)branch=='generate_pseudo_label'
            if self.specificnet_from_bbox and is_training: 
                bias_specific = self.specific_net(specific_feats).mean(dim=0, keepdim=True).unsqueeze(1)  # (n_cls or n_cls+1,  1,  ctx_dim)
                if domain_label == 2: # target
                    self.update_specific_bbox_buffer(bias_specific, self.specific_ema_keep_rate) # update specific_bbox_buffer with target domain 
          
            elif self.specificnet_from_bbox and is_inference:
                if self.specificnet_test_with_buffer: # using saved buffer for inference
                    bias_specific = self.specific_bbox_buffer
                else:
                    bias_specific = self.specific_net(specific_feats).mean(dim=0, keepdim=True).unsqueeze(1) 
                
            else: # specificnet from imagewise (not bbox-wise) 
                if self.specificnet_test_with_buffer: # using saved buffer for inference
                    bias_specific = self.specific_bbox_buffer
                else:
                    bias_specific = self.specific_net(specific_feats).mean(dim=0, keepdim=True).unsqueeze(1)  # (1, 1, ctx_dim)
            
            ctx_specific = ctx[:, self.n_agnostic:, :] + bias_specific
            ctx = torch.cat([ctx[:, :self.n_agnostic, :], ctx_specific], dim=1)
            
        if self.learnable_bg:
            prefix = torch.cat([prefix, prefix[:1]], dim=0)
            if self.n_cls == 1:
                suffix = torch.cat([suffix, suffix[:1]], dim=0)
            else:
                suffix = torch.cat([suffix, suffix[2:3]], dim=0)
            
        prompts = torch.cat([
            prefix,
            ctx,
            suffix
            ],
            dim=1,
        )

        return prompts


class ReturnLearnablePrompt(nn.Module):
    def __init__(self, 
                 classnames, 
                 language_encoder, 
                 csc, 
                 agnosticnet: bool = False, 
                 specificnet: bool = False,
                 specificnet_from_bbox: bool = False,
                 specificnet_test_with_buffer: bool = False,
                 metanet_shallow_features: bool = False, 
                 learnable_bg: bool = False
                 ):
        super().__init__()
        
        self.prompt_learner = MSPromptLearner(classnames, 
                                              language_encoder, 
                                              csc, 
                                              agnosticnet, 
                                              specificnet, 
                                              specificnet_from_bbox, 
                                              specificnet_test_with_buffer, 
                                              metanet_shallow_features, 
                                              learnable_bg
        )
        
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(language_encoder)
        
        for name, param in self.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False) # Freeze all except learnable prompt

    def forward(self, domain_label=2, is_inference=False, agnostic_feats=None, specific_feats=None):
        prompts = self.prompt_learner(domain_label, is_inference, agnostic_feats, specific_feats)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        return text_features
