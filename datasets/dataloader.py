# 需要编写get_loaders方法
# Need to implement get_loaders function
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils

import datasets.data_config as data_config
from datasets.CD_dataset import CDDataset

def get_loaders(args):
    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = "val"
    if hasattr(args, "split_val"):
        split_val = args.split_val
    if args.dataset == "CDDataset":
        training_set = CDDataset(
            root_dir=root_dir,
            split=split,
            img_size=args.img_size,
            is_train=True,
            label_transform=label_transform,
        )
        val_set = CDDataset(
            root_dir=root_dir,
            split=split_val,
            img_size=args.img_size,
            is_train=False,
            label_transform=label_transform,
        )
    else:
        raise NotImplementedError(
            "Wrong dataset name %s (choose one from [CDDataset,])" % args.dataset
        )

    datasets = {"train": training_set, "val": val_set}
    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        for x in ["train", "val"]
    }

    return dataloaders
