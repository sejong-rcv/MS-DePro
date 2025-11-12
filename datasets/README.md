# Prepare Datasets

We outline the dataset setup process for Multi-Source Domain Adaptation (MSDA) & Multi-Source Domain Generalization (MSDG).

The following instruction is based on [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md).

## Use Builtin Datasets

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

Detectron2 has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  coco/
  lvis/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

## Download Datasets

A total of 14 datasets are needed in MSDA & MSDG for conducting experiments on both MSDG and MSDA.

- Download datasets for [BDD100K](http://bdd-data.berkeley.edu/), [Cityscapes](https://www.cityscapes-dataset.com/), [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d), [COCO](https://cocodataset.org/#download), and [Synscapes](https://synscapes.on.liu.se/download.html).

- Download datasets for [Diverse Weather Domain Generalization](https://github.com/AmingWu/Single-DGOD?tab=readme-ov-file) and [Real to Artistic Domain Generalization](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets).

## Expected overall dataset structure:
```
# alphabetical order
datasets/
  BDD100K/
    Daytime/
    Night/
    DawnDusk/
  Cityscapes/
  Clipart/
  COCO/
  Comic/
  Daytime_Foggy/
  Daytime_Sunny/
  Dusk_Rainy/
  KITTI/
  Night_Clear/
  Night_Rainy/
  Synscapes/
  VOCdevkit/
    VOC2007/
    VOC2012/
  Watercolor/
```

## Expected dataset structure for [BDD100K](http://bdd-data.berkeley.edu/):
```
BDD100K/
  Daytime/
    Annotations/
    ImageSets/
      Main/
        train.txt
        val.txt
        # train.txt or test.txt, if you use these splits
    JPEGImages/
    depthImages/
  Night/
    Annotations/
    ImageSets/
      Main/
        train.txt
        val.txt
    JPEGImages/
    depthImages/
  DawnDusk/
    Annotations/
    ImageSets/
      Main/
        train.txt
        val.txt
    JPEGImages/
    depthImages/
```

## Expected dataset structure for [Cityscapes](https://www.cityscapes-dataset.com/):
```
Cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
  ImageSets/
    train.txt
    val.txt
    test.txt
    caronly_{train,val}.txt
    # caronly is required for cross-camera training
  depthImages/
    train/
      aachen/
        color.png
```

## Expected dataset structure for [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d):
```
KITTI/
  Annotations/
  ImageSets/
    Main/
      train.txt
      trainval.txt
      val.txt
  JPEGImages/
  depthImages/
```

## Expected dataset structure for [Synscapes](https://synscapes.on.liu.se/download.html):
```
Synscapes/
  img/
    class/
    depth/
    instance/
    rgb/
    rgb-2k/
  meta/

## Expected dataset structure for [Diverse Weather](https://github.com/AmingWu/Single-DGOD?tab=readme-ov-file):

```
Daytime_Sunny/
  Annotations/
  ImageSets/
    Main/
      train.txt
      test.txt
      # train.txt or val.txt, if you use these splits
  JPEGImages/
  depthImages/
  # (optional) depthAnything or depthPro, according to the model

Night_Clear/
  Annotations/
  ImageSets/
    Main/
      train.txt
      trainval.txt
      test.txt
  JPEGImages/
  depthImages/

Daytime_Foggy/
  Annotations/
  ImageSets/
    Main/
      train.txt
      trainval.txt
      test.txt
  JPEGImages/
  depthImages/

Dusk_Rainy/
  Annotations/
  ImageSets/
    Main/
      train.txt
  JPEGImages/
  depthImages/

Night_Rainy/
  Annotations/
  ImageSets/
    Main/
      train.txt
  JPEGImages/
  depthImages/
```

## Expected dataset structure for [Real to Artistic](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets):

```
VOCdevkit/
  VOC2007/
    Annotations/
    ImageSets/
      Main/
        train.txt
        trainval.txt
        test.txt
        # train.txt or val.txt, if you use these splits
    JPEGImages/
    depthImages/
    # (optional) depthAnything or depthPro, according to the model
  VOC2012/
    Annotations/
    ImageSets/
      Main/
        train.txt
        trainval.txt
        test.txt
        # train.txt or val.txt, if you use these splits
    JPEGImages/
    depthImages/

COCO/
  annotations/
    instances_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json

Clipart/
  Annotations/
  ImageSets/
    Main/
      all.txt
      train.txt
      test.txt
  JPEGImages/
  depthImages/

Comic/
  Annotations/
  ImageSets/
    Main/
      all.txt
      extra.txt
      instance_level_annotated.txt
      train.txt
      test.txt
  JPEGImages/
  depthImages/

Watercolor/
  Annotations/
  ImageSets/
    Main/
      all.txt
      extra.txt
      instance_level_annotated.txt
      train.txt
      test.txt
  JPEGImages/
  depthImages/
```
    {1-25000}.json
  depthImages/
    {1-25000}.png
```
