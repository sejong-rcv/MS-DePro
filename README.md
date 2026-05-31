# Multi-Modal Guided Multi-Source Domain Adaptation for Object Detection

This is the official PyTorch implementation of MS-DePro (Under Review).

## Outline

1. [Installation](#Installation)
2. [Datasets](#Datasets)
3. [Model Zoo](#Model-Zoo)
4. [Training](#Training)
5. [Inference](#Inference)
6. [Acknowledgement](#Acknowledgement)

## Installation

See [`INSTALL.md`](docs/INSTALL.md) for detailed installation instructions.

## Datasets

See [`datasets/README.md`](datasets/README.md) for dataset preparation guidelines.

## Model Zoo

See [`MODEL_ZOO.md`](./MODEL_ZOO.md) for our pretrained models.

## Training

Our MS-DePro is built with [Detectron2](https://github.com/facebookresearch/detectron2). See [Getting Started with Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) to learn about basic usage. We provide an example below for training our object detector on MSDA and MSDG settings.

<details>

<summary>
1. Prepare the pretrained RegionCLIP model and set up the dataset.
</summary>
  
- Check [`RegionCLIP`](https://github.com/microsoft/RegionCLIP/blob/main/docs/MODEL_ZOO.md) to 
  - download the pretrained RegionCLIP checkpoint `regionclip_pretrained-cc_rn50.pth` to the folder `./pretrained`, 
  - (optional) download the trained RPN checkpoint `rpn_coco_{48,65,80}.pth` to the folder `./pretrained`.
- Check [`datasets/README.md`](datasets/README.md) to set up dataset.

</details>

<details>

<summary>
2. After preparation, run the following script to train an object detector.
</summary>

```
#!/bin/bash

CONFIG=$1
NUM_GPUS=$2
CUDA_DEVICES=$3
NUM_THREADS=$4
NUM_SRCS=$5
IMG_BATCH=$6
OUTPUT_DIR=$7

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$NUM_THREADS \
    python tools/train_net.py \
    --num-gpus $NUM_GPUS \
    --config $CONFIG \
    MODEL.BACKBONE_WEIGHTS pretrained/regionclip_pretrained-cc_rn50.pth \
    MODEL.RESNETS.OUT_FEATURES "(('res2'), ('res4'))" \
    DATASETS.NUM_SOURCES $NUM_SRCS \
    SOLVER.IMG_PER_BATCH_LABEL $IMG_BATCH SOLVER.IMG_PER_BATCH_UNLABEL $IMG_BATCH \
    OUTPUT_DIR $OUTPUT_DIR
```

For example, to run the `Cross-time` experiment using 4 GPUs, execute the following command:
```
sh dist_train.sh configs/MSDA/cross_time.sh 4 0,1,2,3 8 2 8 output/cross_time
```

</details>

## Inference

We provide an example below for evaluating our object detector on MSDA and MSDG settings.

<details>

<summary>
1. Prepare the trained detector and set up the dataset.
</summary>
  
- Check [`MODEL_ZOO.md`](MODEL_ZOO.md) to 
  - download the trained detector checkpoints to the folder `./output/`.
- Check [`datasets/README.md`](datasets/README.md) to set up dataset.

</details>

<details>

<summary>
2. After preparation, run the following script to evaluate an object detector.
</summary>
  
```
#!/bin/bash

CONFIG=$1
NUM_GPUS=$2
CUDA_DEVICES=$3
NUM_THREADS=$4
NUM_SRCS=$5
WEIGHTS=$6
OUTPUT_DIR=$7

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$NUM_THREADS \
    python tools/train_net.py \
    --eval-only \
    --num-gpus $NUM_GPUS \
    --config $CONFIG \
    MODEL.BACKBONE_WEIGHTS pretrained/regionclip_pretrained-cc_rn50.pth \
    MODEL.WEIGHTS $WEIGHTS \
    MODEL.RESNETS.OUT_FEATURES "(('res2'), ('res4'))" \
    DATASETS.NUM_SOURCES $NUM_SRCS \
    OUTPUT_DIR $OUTPUT_DIR
```

For example, to evaluate the `Cross-time` experiment using a single GPU, execute the following command:
```
sh slurm_test.sh configs/MSDA/cross_time.yaml 1 0 1 2 output/cross_time.pth eval/cross_time
```

</details>

## Acknowledgement
This repository was built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [RegionCLIP](https://github.com/microsoft/RegionCLIP), [ACIA](https://github.com/imatif17/ACIA) and [DAPrompt](https://github.com/LeapLabTHU/DAPrompt). We are grateful for the contributions from our community.
