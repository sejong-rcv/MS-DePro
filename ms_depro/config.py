# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_cfg(cfg):
    """
    Add config.
    """
    _C = cfg

    _C.MODEL.BACKBONE_WEIGHTS = None
    _C.MODEL.BB_RPN_WEIGHTS = None # "pretrained/R-50_coco.pkl"
    _C.MODEL.RESNETS.OUT_FEATURES "(('res2'), ('res4'))"
    _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
    _C.MODEL.RPN.LOSS = "CrossEntropy"
    _C.MODEL.ROI_HEADS.LOSS = "FocalLoss"
    # Use soft NMS if True
    _C.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    # See soft NMS paper for definition of these options
    _C.MODEL.ROI_HEADS.SOFT_NMS_METHOD = "gaussian" # "linear"
    _C.MODEL.ROI_HEADS.SOFT_NMS_SIGMA = 0.5
    # For the linear_threshold we use NMS_THRESH_TEST
    _C.MODEL.ROI_HEADS.SOFT_NMS_PRUNE = 0.001
    _C.MODEL.PIXEL_MEAN_DEPTH = [103.530, 116.280, 123.675]
    _C.MODEL.PIXEL_STD_DEPTH= [1.0, 1.0, 1.0]
    
    _C.SOLVER.IMG_PER_BATCH_LABEL = 8
    _C.SOLVER.IMG_PER_BATCH_UNLABEL = 8
    _C.SOLVER.FACTOR_LIST = (1,)

    _C.DATASETS.TRAIN_LABEL = ("coco_2017_train",)
    _C.DATASETS.TRAIN_UNLABEL = ("coco_2017_train",)
    _C.DATASETS.VAL_LABEL = ("coco_2017_val",)
    _C.DATASETS.NUM_SOURCES = 1

    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.CLASS = (None, )
    _C.DATALOADER.DEPTH = "depthpro" # "depthanything", "depthpro" or others
    
    _C.EMAMODEL = CN()
    _C.EMAMODEL.SUP_CONSIST = True

    _C.INPUT.MIN_SIZE_TEST = 600
    _C.TEST.MS_MODE = "MSDA" # "MSDG"
    _C.TEST.EVALUATOR = "COCO"
    _C.TEST.PERIOD_STUDENT = 1000
    _C.TEST.PERIOD_TEACHER = 1000
    
    # ---------------------------------------------------------------------------- #
    # CLIP options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.CLIP = CN()

    _C.MODEL.CLIP.CROP_REGION_TYPE = "" # options: "GT", "RPN" 
    _C.MODEL.CLIP.BB_RPN_WEIGHTS = None # the weights of pretrained MaskRCNN
    _C.MODEL.CLIP.IMS_PER_BATCH_TEST = 8 # the #images during inference per batch

    _C.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER = False # if True, use the CLIP text embedding as the classifier's weights
    _C.MODEL.CLIP.TEXT_EMB_PATH = None
    _C.MODEL.CLIP.OFFLINE_RPN_CONFIG = None # option: all configs of pretrained RPN
    _C.MODEL.CLIP.NO_BOX_DELTA = False  # if True, during inference, no box delta will be applied to region proposals

    _C.MODEL.CLIP.BG_CLS_LOSS_WEIGHT = None # if not None, it is the loss weight for bg regions
    _C.MODEL.CLIP.ONLY_SAMPLE_FG_PROPOSALS = False  # if True, during training, ignore all bg proposals and only sample fg proposals
    _C.MODEL.CLIP.MULTIPLY_RPN_SCORE = False  # if True, during inference, multiply RPN scores with classification scores
    _C.MODEL.CLIP.VIS = False # if True, when visualizing the object scores, we convert them to the scores before multiplying RPN scores

    _C.MODEL.CLIP.OPENSET_TEST_NUM_CLASSES = None  # if an integer, it is #all_cls in test
    _C.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH = None # if not None, enables the openset/zero-shot training, the category embeddings during test

    _C.MODEL.CLIP.CLSS_TEMP = 0.01 # normalization + dot product + temperature
    _C.MODEL.CLIP.RUN_CVPR_OVR = False # if True, train CVPR OVR model with their text embeddings
    _C.MODEL.CLIP.FOCAL_SCALED_LOSS = None # if not None (float value for gamma), apply focal loss scaling idea to standard cross-entropy loss

    _C.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH = None # the threshold of NMS in offline RPN
    _C.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST = None # the number of region proposals from offline RPN
    _C.MODEL.CLIP.PRETRAIN_IMG_TXT_LEVEL = True # if True, pretrain model using image-text level matching
    _C.MODEL.CLIP.PRETRAIN_ONLY_EOT = False # if True, use end-of-token emb to match region features, in image-text level matching
    _C.MODEL.CLIP.PRETRAIN_RPN_REGIONS = None # if not None, the number of RPN regions per image during pretraining
    _C.MODEL.CLIP.PRETRAIN_SAMPLE_REGIONS = None # if not None, the number of regions per image during pretraining after sampling, to avoid overfitting
    _C.MODEL.CLIP.GATHER_GPUS = False # if True, gather tensors across GPUS to increase batch size
    _C.MODEL.CLIP.GRID_REGIONS = False # if True, use grid boxes to extract grid features, instead of object proposals
    _C.MODEL.CLIP.CONCEPT_POOL_EMB = None # if not None, it provides the file path of embs of concept pool and thus enables region-concept matching
    _C.MODEL.CLIP.CONCEPT_THRES = None # if not None, the threshold to filter out the regions with low matching score with concept embs, dependent on temp (default: 0.01)

    _C.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED = False # if True, use large-scale jittering (LSJ) pretrained RPN
    _C.MODEL.CLIP.TEACHER_RESNETS_DEPTH = 50 # the type of visual encoder of teacher model, sucha as ResNet 50, 101, 200 (a flag for 50x4)
    _C.MODEL.CLIP.TEACHER_CONCEPT_POOL_EMB = None # if not None, it uses the same concept embedding as student model; otherwise, uses a seperate embedding of teacher model
    _C.MODEL.CLIP.TEACHER_POOLER_RESOLUTION = 14 # RoIpooling resolution of teacher model

    _C.MODEL.CLIP.TEXT_EMB_DIM = 1024 # the dimension of precomputed class embeddings
    _C.INPUT_DIR = "./datasets/custom_images" # the folder that includes the images for region feature extraction
    _C.MODEL.CLIP.GET_CONCEPT_EMB = False # if True (extract concept embedding), a language encoder will be created

    # ---------------------------------------------------------------------------- #
    # Teacher-Student Model options
    # ---------------------------------------------------------------------------- #
    _C.TSMODEL = CN()

    # Output dimension of the MLP projector after `res5` block
    _C.TSMODEL.MLP_DIM = 128

    # Teacher-student training
    _C.TSMODEL.BURN_UP_STEP = 80000
    _C.TSMODEL.BBOX_THRESHOLD = 0.9
    _C.TSMODEL.PSEUDO_BBOX_SAMPLE = "thresholding"
    _C.TSMODEL.TEACHER_UPDATE_ITER = 1
    _C.TSMODEL.EMA_KEEP_RATE = 0.0
    _C.TSMODEL.LOSS_WEIGHT_TYPE = "standard"
    _C.TSMODEL.SUP_LOSS_WEIGHT = 1.0
    _C.TSMODEL.UNSUP_LOSS_WEIGHT = 1.0
    
    # ---------------------------------------------------------------------------- #
    # Multi-Modal Guided Learnable Prompt options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.CLIP.LEARNABLE_PROMPT = True # Set learnable text prompts if true.
    _C.MODEL.CLIP.CSC = False # Convert domain-specific tokens to class-specific if true.
    _C.MODEL.CLIP.AGNOSTICNET = True
    _C.MODEL.CLIP.AGNOSTICNET_FROM_BBOX = False
    _C.MODEL.CLIP.SPECIFICNET = True
    _C.MODEL.CLIP.SPECIFICNET_FROM_BBOX = True
    _C.MODEL.CLIP.SPECIFICNET_INCLUDE_BG = True
    _C.MODEL.CLIP.SPECIFICNET_TEST_WITH_BUFFER = False
    _C.MODEL.CLIP.METANET_SHALLOW_FEATURES = True
    _C.MODEL.CLIP.LEARNABLE_BG = True
