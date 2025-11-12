import copy
import time
import pickle
import logging
import os.path as osp
import numpy as np
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.events import EventStorage
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.engine import TrainerBase, DefaultTrainer, hooks
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.evaluation import verify_results, DatasetEvaluators

from ms_depro.data import MetadataCatalog
from ms_depro.data.build import (
    build_ms_detection_train_loader,
    build_detection_test_loader
)
from ms_depro.data.dataset_mapper import MSDatasetMapper
from ms_depro.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ms_depro.checkpoint.detection_ts_checkpoint import DetectionTSCheckpointer
from ms_depro.solver.build import build_lr_scheduler
from ms_depro.evaluation import COCOEvaluator


class MSTrainer(DefaultTrainer):
    def __init__(self, cfg):
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        
        all_source = []
        for s_i in range(cfg.DATASETS.NUM_SOURCES):
            source = self.build_train_loader(cfg, ind=s_i, is_source=True)
            all_source.append(source)
        target = self.build_train_loader(cfg, ind=s_i+1, is_source=False)
        
        self.all_source = all_source
        self.target = target

        model = self.build_model(cfg)
        if cfg.MODEL.BACKBONE_WEIGHTS is not None:
            model = self._load_pretrained(model=model, 
                                          cfg=cfg,
                                          load_depth=(cfg.DATALOADER.DEPTH != "RGB"),
                                          load_rpn=(cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None))
        
        optimizer = build_optimizer(cfg, model)
        # Teacher model is updated from the student
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, 
                device_ids=[comm.get_local_rank()], 
                broadcast_buffers=False, 
                find_unused_parameters=True
            )
        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, self.all_source[0], optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        model_ts = EnsembleTSModel(model_teacher, model) # teacher-student framework
        self.checkpointer = DetectionTSCheckpointer(
            model_ts,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        
        self.cfg = cfg
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER

        self.data_loader_iter_sources = [None] * len(cfg.DATASETS.TRAIN_LABEL)
        self.source_data = self.data_loader_iter_sources # optional alias
        self.data_loader_iter_target = None
        self.register_hooks(self.build_hooks())
        self.all_unlabel_data = None


    def _data_loader_iter_s(self, i):
        if self.data_loader_iter_sources[i] is None:
            self.data_loader_iter_sources[i] = iter(self.all_source[i])
        return self.data_loader_iter_sources[i]
    
    @property
    def _data_loader_iter_t(self):
        if self.data_loader_iter_target is None:
            self.data_loader_iter_target = iter(self.target)
        return self.data_loader_iter_target
    
    @classmethod
    def build_train_loader(cls, cfg, ind=0, is_source=True):
        map_func = MSDatasetMapper(cfg, is_train=True)
        return build_ms_detection_train_loader(cfg, map_func, ind, is_source)
    
    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("model: {}".format(model))
        return model
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)
    
    def _load_pretrained(
        self, model, cfg, load_depth=True, load_rpn=False
    ):
        _weights = {}
        
        # CLIP ResNet
        if "CLIP" in cfg.MODEL.BACKBONE["NAME"].upper():
            pretrained = torch.load(cfg.MODEL.BACKBONE_WEIGHTS, map_location="cpu")["model"]
            for n, wei in pretrained.items():
                if n.startswith('backbone'):
                    _weights[n] = wei
                    if load_depth:
                        _weights[n.replace('backbone', 'backbone_d')] = wei
        else: # Standard ResNet
            if cfg.MODEL.BACKBONE["NAME"] == "build_resnet_backbone":
                with open(cfg.MODEL.BACKBONE_WEIGHTS, 'rb') as c:
                    resnet_backbone = pickle.load(c)
                for n, wei in resnet_backbone['model'].items():
                    if n.startswith('backbone'):
                        _weights[n] = torch.from_numpy(wei)
                        if load_depth:
                            _weights[n.replace('backbone', 'backbone_d')] = torch.from_numpy(wei)
            else:
                raise ValueError("Unknown backbone name: {}".format(cfg.MODEL.BACKBONE["NAME"]))
        
        # You can transfer pretrained RPN weights
        # However, we are not used this weights for our paper.
        if load_rpn:
            assert cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None
            
            rpn_weights = torch.load(cfg.MODEL.CLIP.BB_RPN_WEIGHTS)['model'] # coco RPN
            for n, wei in rpn_weights.items():
                if n.startswith('proposal_generator'):
                    _weights[n] = wei
                    if load_depth:
                        _weights[n.replace('proposal_generator', 'proposal_generator_d')] = wei

        model.load_state_dict(_weights, strict=False)
        del _weights
        
        return model
    
    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume:
            self.start_iter = checkpoint.get("iteration", -1) + 1
        if isinstance(self.model, DistributedDataParallel):
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
        self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "None evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)
          
    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results
        
    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))
        self.iter = start_iter
        self.max_iter = max_iter
        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def threshold_boxes(
        self, proposal_boxes_inst, thres=0.5, proposal_type="roih"
    ):
        if proposal_type == "rpn":
            valid_map = proposal_boxes_inst.objectness_logits > thres
            image_shape = proposal_boxes_inst.image_size
            new_proposal_inst = Instances(image_shape)
            new_bbox_loc = proposal_boxes_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_boxes_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_boxes_inst.scores > thres

            image_shape = proposal_boxes_inst.image_size
            new_proposal_inst = Instances(image_shape)

            new_bbox_loc = proposal_boxes_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_boxes_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_boxes_inst.scores[valid_map]
        else:
            raise ValueError("Unknown proposal type: {}".format(propoal_type))

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, pseudo_label_method="thresholding"
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            if pseudo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods: {}".format(pseudo_label_method))
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        
        return list_instances, num_proposal_output

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
            
        return unlabled_data
    
    def teacher_predictions(
        self, unlabel_data_weak, unlabel_data_strong
    ):
        with torch.no_grad():
            proposals_rpn_unsup_weak, proposals_roih_unsup_weak = \
                self.model_teacher(unlabel_data_weak, branch="generate_pseudo_label")

        cur_threshold = self.cfg.TSMODEL.BBOX_THRESHOLD

        joint_proposal_dict = {}
        joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_weak
        pesudo_proposals_rpn_unsup_weak, nun_pseudo_bbox_rpn = self.process_pseudo_label(
            proposals_rpn_unsup_weak, cur_threshold, "rpn", "thresholding")

        joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_weak
        pesudo_proposals_roih_unsup_weak, _ = self.process_pseudo_label(
            proposals_roih_unsup_k, cur_threshold, "roih", "thresholding")
        joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_weak

        _unlabel_data_strong = self.add_label(
            unlabel_data_strong, joint_proposal_dict["proposals_pseudo_roih"])
        
        return _unlabel_data_strong
    
    def remove_label(self, label_data):   
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
                
        return label_data
            
    def run_step(self):
        data_t = next(self._data_loader_iter_t)
        unlabel_data_strong, unlabel_data_weak = data_t

        total_losses = 0
        for i, _ in enumerate(self.source_data):
            assert self.model.training, "model was changed to eval"
           
            start_time = time.perf_counter()
            data_s = next(self._data_loader_iter_s(i))
            label_data_strong, label_data_weak = data_s
            data_time = time.perf_counter() - start_time
           
            # Labeled source-only training
            if (self.iter < self.cfg.TSMODEL.BURN_UP_STEP):
                # This leads to stable training of the model.
                label_data_strong.extend(label_data_weak)
                start_time = time.perf_counter()
                record_dict = self.model(batched_inputs=label_data_strong, 
                                         branch="supervised_source", 
                                         source_label=i)
                forward_time = time.perf_counter() - start_time
                loss_dict = {}
                for key in record_dict.keys():
                    if key[:4] == "loss":
                        loss_dict[key] = record_dict[key] * 1.
                losses = sum(loss_dict.values())
            else: # Target adaptation
                if (i==0) and (self.iter == self.cfg.TSMODEL.BURN_UP_STEP):
                    self._update_teacher_model(keep_rate=0.00)
                elif (i==0) and ((
                    self.iter - self.cfg.TSMODEL.BURN_UP_STEP
                ) % self.cfg.TSMODEL.TEACHER_UPDATE_ITER == 0):
                    self._update_teacher_model(
                        keep_rate=self.cfg.TSMODEL.EMA_KEEP_RATE
                    )
                # Pseudo-label for teacher model
                if (i==0):
                    unlabel_data_strong = self.remove_label(unlabel_data_strong)
                    unlabel_data_weak = self.remove_label(unlabel_data_weak)
                    self.all_unlabel_data = self.teacher_predictions(
                        unlabel_data_weak, unlabel_data_strong
                    )
                    
                start_time = time.perf_counter()
                record_dict = {}
                # Labeled source training
                label_data_strong.extend(label_data_weak)
                record_dict.update(
                    self.model(batched_inputs=label_data_strong,
                               branch="supervised_source", 
                               source_label=i)
                )
                # Pseudo target training
                record_all_unlabel_data = self.model(batched_inputs=self.all_unlabel_data,
                                                     branch="pseudo_training_target",
                                                     source_label=i)
                forward_time = time.perf_counter() - start_time
                new_record_all_unlabel_data = {}
                for key in record_all_unlabel_data.keys():
                    new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
                record_dict.update(new_record_all_unlabel_data)
                
                loss_dict = {}
                for key in record_dict.keys():
                    if key.startswith("loss"):
                        if key in ["loss_rpn_loc_pseudo", "loss_box_reg_pseudo"]:
                            loss_dict[key] = record_dict[key] * 0.
                        elif key[-6:] == "pseudo": # unsupervised loss
                            loss_dict[key] = (record_dict[key] * self.cfg.TSMODEL.UNSUP_LOSS_WEIGHT)
                        else: # supervised loss
                            loss_dict[key] = (record_dict[key] * self.cfg.TSMODEL.SUP_LOSS_WEIGHT)
                losses = sum(loss_dict.values())
            total_losses += losses
            
        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        metrics_dict["forward_time"] = forward_time

        self._write_metrics(metrics_dict)
        self.optimizer.zero_grad()
        total_losses.backward()
        self.optimizer.step()
            
    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)
            
            if "forward_time" in all_metrics_dict[0]:
                forward_time = np.max([x.pop("forward_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("forward_time", forward_time)

            # Average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        update_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                update_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))
            
    @torch.no_grad()
    def _copy_main_model(self):
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())
            
    def build_hooks(self):
        cfg = self.cfg.clone()

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.PERIOD_STUDENT,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )
            
        def test_and_save_results_teacher():
            if self.iter < self.cfg.TSMODEL.BURN_UP_STEP:
                return
            self._last_eval_results_teacher = self.test(
                self.cfg, 
                self.model_teacher
            )
            return self._last_eval_results_teacher

        def test_and_save_results_student():
            if self.iter > self.cfg.TSMODEL.BURN_UP_STEP:
                return
            self._last_eval_results_student = self.test(
                self.cfg, 
                self.model
            )
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student
        
        ret.append(hooks.EvalHook(cfg.TEST.PERIOD_STUDENT,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.PERIOD_TEACHER,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
