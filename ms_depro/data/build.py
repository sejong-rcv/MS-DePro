# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import logging
import operator
import itertools
from tabulate import tabulate
from termcolor import colored

from detectron2.utils.comm import get_world_size
from detectron2.data.common import (
    DatasetFromList,
    MapDataset
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
    filter_images_with_only_crowd_annotations,
    filter_images_with_few_keypoints
)

from ms_depro.data.common import AspectRatioGroupedDatasetTwoCrop
from ms_depro.data.catalog import DatasetCatalog, MetadataCatalog


"""
This file contains the default logic to build a dataloader for training or testing.
"""     
def print_instances_class_histogram(dataset_dicts, class_names, dataset_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
        dataset_names (list[str]): list of dataset names.
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = np.asarray(
            [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=int
        )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="center",
        stralign="center",
    )
    # we manually set caller_module as `detectron2.data.build` (refer to function `log_first_n`)
    logging.getLogger('detectron2.data.build').log(
        logging.INFO, 
        "Distribution of instances among all {} categories on {}:\n".format(num_classes, dataset_names[0])
        + colored(table, "cyan"),
        )


def get_detection_dataset_dicts(
    cfg, dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None, check_consistency=True,
    ):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names), dataset_names

    available_datasets = DatasetCatalog.keys()
    names_set = set(dataset_names)
    if not names_set.issubset(available_datasets):
        logger = logging.getLogger(__name__)
        logger.warning(
            "The following dataset names are not registered in the DatasetCatalog: "
            f"{names_set - available_datasets}. "
            f"Available datasets are {available_datasets}"
        )
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    
    if isinstance(dataset_dicts[0], torch.utils.data.Dataset):
        if len(dataset_dicts) > 1:
            # ConcatDataset does not work for iterable style dataset.
            # We could support concat for iterable as well, but it's often
            # not a good idea to concat iterables anyway.
            return torch.utils.data.Dataset.ConcatDataset(dataset_dicts)
        return dataset_dicts[0]

    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)
    if check_consistency and has_instances:
        try:
            class_names = MetadataCatalog.get(dataset_names[0]).thing_classes
            print_instances_class_histogram(dataset_dicts, class_names, dataset_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts


def build_ms_batch_dataloader(
    dataset, sampler, total_batch_size, *, aspect_ratio_grouping=False, num_workers=0
    ):
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size

    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # Yield individual elements (not batched)
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedDatasetTwoCrop(
            data_loader, batch_size
        )
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING=False is not supported yet")


def build_ms_detection_train_loader(
    cfg, map_func=None, ind=0, is_source=True
    ):
        if is_source: # (labeled) source dataset
            label_dicts = get_detection_dataset_dicts(
                cfg,
                dataset_names=cfg.DATASETS.TRAIN_LABEL[ind],
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
            )
            label_dataset = DatasetFromList(label_dicts, copy=False)
            label_dataset = MapDataset(label_dataset, map_func)

            sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
            logger = logging.getLogger(__name__)
            logger.info("Using training sampler {}".format(sampler_name))
            if sampler_name == "TrainingSampler":
                label_sampler = TrainingSampler(len(label_dataset))
            elif sampler_name == "RepeatFactorTrainingSampler":
                raise NotImplementedError("{} not yet supported.".format(sampler_name))
            else:
                raise ValueError("Unknown training sampler: {}".format(sampler_name))
            return build_ms_batch_dataloader(
                label_dataset,
                label_sampler,
                cfg.SOLVER.IMG_PER_BATCH_LABEL,
                aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
            )

        else: # (unlabeled) target dataset
            unlabel_dicts = get_detection_dataset_dicts(
                cfg,
                dataset_names=cfg.DATASETS.TRAIN_UNLABEL,
                filter_empty=False,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
            )
            unlabel_dataset = DatasetFromList(unlabel_dicts, copy=False)
            unlabel_dataset = MapDataset(unlabel_dataset, map_func)
            sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
            logger = logging.getLogger(__name__)
            logger.info("Using training sampler {}".format(sampler_name))
            if sampler_name == "TrainingSampler":
                unlabel_sampler = TrainingSampler(len(unlabel_dataset))
            elif sampler_name == "RepeatFactorTrainingSampler":
                raise NotImplementedError("{} not yet supported.".format(sampler_name))
            else:
                raise ValueError("Unknown training sampler: {}".format(sampler_name))
            return build_ms_batch_dataloader(
                unlabel_dataset,
                unlabel_sampler,
                cfg.SOLVER.IMG_PER_BATCH_UNLABEL,
                aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
            )


def build_detection_test_loader(cfg, dataset_names, map_func=None):
    dataset_dicts = get_detection_dataset_dicts(
        cfg,
        dataset_names=[dataset_names],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ] if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    dataset = DatasetFromList(dataset_dicts)
    if map_func is None:
        map_func = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, map_func)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader
