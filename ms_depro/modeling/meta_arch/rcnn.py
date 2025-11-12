# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from typing import List, Dict, Tuple, Optional

from detectron2.config import configurable
from detectron2.utils.events import get_event_storage
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from ms_depro.modeling.backbone import Backbone, build_backbone
from ms_depro.modeling.backbone.clip_backbone import build_clip_language_encoder
from ms_depro.modeling.proposal_generator import build_proposal_generator
from ms_depro.modeling.roi_heads import build_roi_heads

from ..fusion import avg_proposal, weighted_proposal
from ..postprocessing import detector_postprocess


__all__ = ["MSCLIPRCNN"]


@META_ARCH_REGISTRY.register()
class MSCLIPRCNN(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        backbone_d: Backbone,
        proposal_generator: nn.Module,
        proposal_generator_d: nn.Module,
        language_encoder: nn.Module,
        roi_heads: nn.Module,
        use_clip_c4: bool,
        use_clip_attpool: bool,
        target_label:int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            backbone_d: a backbone module (for depth map), must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            proposal_generator_d: a module (for depth map) that generates proposals using backbone_d features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
            use_clip_c4: if True, use C4 mode where roi_head uses the last resnet layer from backbone
            use_clip_attpool: if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool
        """
        super().__init__()
        self.backbone = backbone
        self.backbone_d = backbone_d
        self.proposal_generator = proposal_generator
        self.proposal_generator_d = proposal_generator_d
        self.lang_encoder = language_encoder
        self.roi_heads = roi_heads
        
        self.use_clip_c4 = use_clip_c4
        self.use_clip_attpool = use_clip_attpool
        
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        
        self.target_label = target_label
        
    @property
    def device(self):
        return self.pixel_mean.device
    
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_d = build_backbone(cfg)
        # build language encoder
        if cfg.MODEL.CLIP.GET_CONCEPT_EMB:
            language_encoder = build_clip_language_encoder(cfg)
        else:
            language_encoder = None
        return {
            "backbone": backbone,
            "backbone_d": backbone_d,
            "language_encoder": language_encoder,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "proposal_generator_d": build_proposal_generator(cfg, backbone_d.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "use_clip_c4": cfg.MODEL.BACKBONE.NAME == "build_clip_resnet_backbone",
            "use_clip_attpool": cfg.MODEL.ROI_HEADS.NAME in ['CLIPRes5ROIHeads', 'CLIPStandardROIHeads'] and cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
            "target_label": cfg.DATASETS.NUM_SOURCES,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD
        }

    def _preprocess_single_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images (RGB).
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.input_format == 'BGR':
            images = [x[[2,1,0],:,:] for x in images]
        images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
    
    def _preprocess_multiple_images(self, batched_inputs):
        """
        Normalize, pad and batch the input images (RGB and Depth map).
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.input_format == 'BGR':
            images = [x[[2,1,0],:,:] for x in images]
        images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        depth_maps = [x["depth_map"].to(self.device) for x in batched_inputs]
        if self.input_format == 'BGR':
            depth_maps = [x[[2,1,0],:,:] for x in depth_maps]
        depth_maps = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in depth_maps]
        depth_maps = ImageList.from_tensors(depth_maps, self.backbone.size_divisibility)
        return images, depth_maps
    
    def _refine_proposals(
        self, 
        proposals: list,
        threshold: float =0.5, 
        method: str ="s-avg"
    ):
        """
        Args:
            proposals (list): image meta information and proposal lists
                              (i.e. [Instances(num_instances, image_height, image_width, 
                                    fields=[proposal_boxes, objectness_logits]])
            method (str): One of "avg", "s-avg", "argmax"
        Returns:
            proposals (list)
        """
        for proposal_i, proposal in enumerate(proposals):
            device = proposals[proposal_i].proposal_boxes.device
            
            logits = proposal.objectness_logits
            boxes = proposal.proposal_boxes
            order = np.arange(0, len(logits)) # logits already sorted
            
            while order.size > 0:
                i = order[0]
                ovr = pairwise_iou(boxes[i:i+1], boxes[order[1:]])[0].cpu()
                inds = np.where(ovr <= threshold)[0]
                match = np.where(ovr > threshold)[0]
                match_ind = order[match+1]
                match_logits = logits[match_ind].tolist()
                match_proposal_box = boxes[match_ind].tensor.tolist()
                logits_i = logits[i].item()
                proposal_box_i = boxes[i:i+1].tensor.tolist()
                
                if len(match_logits):
                    match_logits.append(logits_i)
                    match_proposal_box += proposal_box_i
                match_prob = torch.sigmoid(torch.tensor(match_logits))
                if method == 'avg':                
                    out_proposal_box = avg_proposal(match_proposal_box)
                elif method == 's-avg':
                    out_proposal_box = weighted_proposal(
                        match_proposal_box, match_prob.tolist()
                    )
                elif method == 'argmax':      
                    if not len(match_prob):
                        out_proposal_box = []
                    else:                        
                        max_score_id = np.argmax(match_prob.tolist())
                        out_proposal_box = match_proposal_box[max_score_id]    
                if not len(out_proposal_box):
                    order = order[inds + 1]
                    continue
                else:
                    proposals[proposal_i].proposal_boxes.tensor[i] = torch.tensor(out_proposal_box, device=device)
                order = order[inds + 1]
        return proposals
    
    @staticmethod
    def _select_top_k_proposals(
        proposals: list, 
        proposals_d: list, 
        is_top_k: bool = False
    ):
        assert len(proposals) == len(proposals_d), "The number of proposals doesn't match"
        
        topk_proposals = []
        for instances, instances_d in zip(proposals, proposals_d):
            proposals_b = torch.cat((instances.get('proposal_boxes').tensor, 
                                     instances_d.get('proposal_boxes').tensor), dim=0)
            objectness_logits_b = torch.cat((instances.get('objectness_logits'), 
                                             instances_d.get('objectness_logits')), dim=0)  
            # By default, Faster R-CNN yield 2000 proposals, If top_k is True,
            # we select 2000 proposals, which are half of the combined set.
            if is_top_k:
                num_instances = instances.get('proposal_boxes').tensor.shape[0]
                num_instances_d = instances_d.get('proposal_boxes').tensor.shape[0]
                
                top_k = (num_instances + num_instances_d) // 2
                indices_desc = torch.argsort(objectness_logits_b, descending=True)
                sorted_proposals = proposals_b[indices_desc][:top_k]
                sorted_objectness_logits = objectness_logits_b[indices_desc][:top_k]
            else:
                indices_desc = torch.argsort(objectness_logits_b, descending=True)
                sorted_proposals = proposals_b[indices_desc]
                sorted_objectness_logits = objectness_logits_b[indices_desc]
            topk_proposals.append(
                Instances(
                    image_size=(instances._image_size[0], instances._image_size[1]),
                    proposal_boxes=Boxes(sorted_proposals),
                    objectness_logits=sorted_objectness_logits)
                )        
        return topk_proposals
    
    def forward(
        self, 
        batched_inputs: List[Dict[str, torch.Tensor]], 
        branch: str ="supervised_source",
        source_label: int = 0,
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images, depth_maps = self._preprocess_multiple_images(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
            
        features = self.backbone(images.tensor)
        features_d = self.backbone_d(depth_maps.tensor)
        
        losses = {}
        
        # Training with labeled source data.
        if branch == "supervised_source":
            proposals_rpn, proposal_losses = self.proposal_generator(images, features, gt_instances)
            proposals_rpn_d, proposal_losses_d = self.proposal_generator_d(depth_maps, features_d, gt_instances)
            topk_proposals = self._select_top_k_proposals(proposals_rpn, proposals_rpn_d, is_top_k=True)
            
            losses.update(proposal_losses)
            losses.update(proposal_losses_d)

            # See https://github.com/microsoft/RegionCLIP
            # Given the proposals, crop region features from 2D image features and classify the regions
            if self.use_clip_c4: # use C4 + resnet weights from CLIP
                if self.use_clip_attpool: # use attention pooling from CLIP to match dimension
                    detector_losses = self.roi_heads(images, 
                                                     features, 
                                                     topk_proposals, 
                                                     gt_instances, 
                                                     compute_loss=True, 
                                                     branch=branch, 
                                                     res5=self.backbone.layer4, 
                                                     attnpool=self.backbone.attnpool,
                                                     domain_label=self.target_label, 
                                                     agnostic_feats=features_d['res2'])
                else:
                    detector_losses = self.roi_heads(images, 
                                                     features, 
                                                     topk_proposals, 
                                                     gt_instances, 
                                                     compute_loss=True, 
                                                     branch=branch, 
                                                     res5=self.backbone.layer4,
                                                     domain_label=self.target_label, 
                                                     agnostic_feats=features_d['res2'])
            else:  # regular detector setting
                if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                    detector_losses = self.roi_heads(images, 
                                                     features, 
                                                     topk_proposals, 
                                                     gt_instances, 
                                                     compute_loss=True, 
                                                     branch=branch, 
                                                     attnpool=self.backbone.bottom_up.attnpool)
                else: # use mean pool
                    detector_losses = self.roi_heads(images, 
                                                     features, 
                                                     topk_proposals, 
                                                     gt_instances, 
                                                     compute_loss=True, 
                                                     branch=branch)
                    
            losses.update(detector_losses)
            
            # visualize
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch + "_RGB")
                    self.visualize_training(batched_inputs, proposals_rpn_d, branch + "_DEPTH_MAP")
    
            return losses
        
        # Generate pseudo-labels on unlabeled target data
        elif branch == "generate_pseudo_label":
            proposals_rpn, _ = self.proposal_generator(images, features, gt_instances=None, compute_loss=False)
            proposals_rpn_d, _ = self.proposal_generator_d(depth_maps, features_d, gt_instances=None, compute_loss=False)
            topk_proposals = self._select_top_k_proposals(proposals_rpn, proposals_rpn_d, is_top_k=True)
            topk_proposals = self._refine_proposals(topk_proposals, threshold=0.5, method="s-avg")
            
            if self.use_clip_c4: # use C4 + resnet weights from CLIP
                if self.use_clip_attpool: # use attention pooling from CLIP to match dimension
                    proposals_roih = self.roi_heads(images, 
                                                    features, 
                                                    topk_proposals,
                                                    compute_loss=False, 
                                                    branch=branch, 
                                                    res5=self.backbone.layer4,
                                                    attnpool=self.backbone.attnpool,
                                                    domain_label=self.target_label, 
                                                    agnostic_feats=features_d['res2'])
                else:
                    proposals_roih = self.roi_heads(images, 
                                                    features, 
                                                    topk_proposals,
                                                    compute_loss=False, 
                                                    branch=branch, 
                                                    res5=self.backbone.layer4,
                                                    domain_label=self.target_label, 
                                                    agnostic_feats=features_d['res2'])
            else:  # regular detector setting
                if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                    proposals_roih = self.roi_heads(images, 
                                                    features, 
                                                    topk_proposals,
                                                    compute_loss=False, 
                                                    branch=branch, 
                                                    attnpool=self.backbone.bottom_up.attnpool)
                else: # use mean pool
                    proposals_roih  = self.roi_heads(images, 
                                                     features, 
                                                     topk_proposals, 
                                                     compute_loss=False, 
                                                     branch=branch)
            return topk_proposals, proposals_roih
        
        # Training with pseudo-labeled target data
        elif branch == "pseudo_training_target":
            proposals_rpn, proposal_losses = self.proposal_generator(images, features, gt_instances)
            proposals_rpn_d, proposal_losses_d = self.proposal_generator_d(depth_maps, features_d, gt_instances)
            proposal_losses_d = {k.replace('rpn', 'rpn_d'):v for k,v in proposal_losses_d.items()}
            topk_proposals = self._select_top_k_proposals(proposals_rpn, proposals_rpn_d, is_top_k=True)

            losses.update(proposal_losses)
            losses.update(proposal_losses_d)
            
            if self.use_clip_c4: # use C4 + resnet weights from CLIP
                if self.use_clip_attpool: # use attention pooling from CLIP to match dimension
                    detector_losses = self.roi_heads(images, 
                                                     features, 
                                                     topk_proposals, 
                                                     gt_instances, 
                                                     compute_loss=True, 
                                                     branch=branch, 
                                                     res5=self.backbone.layer4, 
                                                     attnpool=self.backbone.attnpool,
                                                     domain_label=self.target_label, 
                                                     agnostic_feats=features_d['res2'])
                else:
                    detector_losses = self.roi_heads(images, 
                                                     features, 
                                                     topk_proposals, 
                                                     gt_instances, 
                                                     compute_loss=True, 
                                                     branch=branch, 
                                                     res5=self.backbone.layer4,
                                                     domain_label=self.target_label, 
                                                     agnostic_feats=features_d['res2'])
            else:  # regular detector setting
                if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                    detector_losses = self.roi_heads(images, 
                                                     features, 
                                                     topk_proposals, 
                                                     gt_instances, 
                                                     compute_loss=True, 
                                                     branch=branch, 
                                                     attnpool=self.backbone.bottom_up.attnpool)
                else: # use mean pool
                    detector_losses = self.roi_heads(images, features, topk_proposals, gt_instances, compute_loss=True, branch=branch)
                    
            losses.update(detector_losses)
            
            # visualize
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch + "_RGB")
                    self.visualize_training(batched_inputs, proposals_rpn_geo, branch + "_DEPTH_MAP")

            return losses
        else: 
            raise NotImplementedError()
        
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self._preprocess_single_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            if self.use_clip_c4: # use C4 + resnet weights from CLIP
                if self.use_clip_attpool: # use attention pooling from CLIP to match dimension
                    results = self.roi_heads(images, 
                                             features, 
                                             proposals, 
                                             targets=None,
                                             compute_loss=False, 
                                             branch="target_inference", 
                                             res5=self.backbone.layer4, 
                                             attnpool=self.backbone.attnpool,
                                             domain_label=self.target_label)
                else:
                    results = self.roi_heads(images, 
                                             features,
                                             proposals, 
                                             targets=None, 
                                             compute_loss=False, 
                                             branch="target_inference", 
                                             res5=self.backbone.layer4,
                                             domain_label=self.target_label)
            else:  # regular detector setting
                if self.use_clip_attpool: # use att_pool from CLIP to match dimension
                    results = self.roi_heads(images, 
                                             features, 
                                             proposals, 
                                             targets=None, 
                                             compute_loss=False, 
                                             branch=None, 
                                             attnpool=self.backbone.bottom_up.attnpool)
                else: # use mean pool
                    results  = self.roi_heads(images, 
                                              features, 
                                              proposals, 
                                              targets=None, 
                                              compute_loss=False, 
                                              branch=None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return MSCLIPRCNN._postprocess(results, batched_inputs)
        else:
            return results
        
    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(
            instances, batched_inputs):
            height = input_per_image["height"]  # original image size, before resizing
            width = input_per_image["width"]  # original image size, before resizing
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    
    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data.detection_utils import convert_image_to_rgb

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["depth_map"] if "DEPTH_MAP" in branch else input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes.tensor.cpu())
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: Ground-Truth "
                + branch
                + " Right: Predicted Proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
