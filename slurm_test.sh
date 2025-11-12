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
