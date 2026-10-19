# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
from config import paths
import torch

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from utils.augment import new_data_aug_generator


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(
            root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    if is_train:
        transform = new_data_aug_generator(args)
    else:
        transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def load_imagenet(
        data_set='IMNET',
        datapath=paths.IMAGENET_PATH,
        input_size=224,
        color_jitter=.3,
        aa='rand-m9-mstd0.5-inc1',
        train_interpolation='bicubic',
        reprob=.25,
        remode='pixel',
        recount=1,
        eval_crop_ratio=0.875,
        batch_size=128,
        num_workers=16,
        debug=False):
    class Args:
        pass
    args = Args()
    args.data_set = data_set
    args.data_path = datapath
    args.input_size = input_size
    args.color_jitter = color_jitter
    args.aa = aa
    args.train_interpolation = train_interpolation
    args.reprob = reprob
    args.remode = remode
    args.recount = recount
    args.eval_crop_ratio = eval_crop_ratio
    train_dataset, _ = build_dataset(is_train=True, args=args)
    val_dataset, _ = build_dataset(is_train=False, args=args)

    if debug:
        train_dataset = Subset(train_dataset, indices=torch.randperm(len(train_dataset))[:4000])
        val_dataset = Subset(val_dataset, indices=torch.randperm(len(val_dataset))[:1000])

    train_dataset = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    val_dataset = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataset, val_dataset, val_dataset


def load_cifar100(
        data_set='CIFAR',
        datapath=paths.DATA_PATH,
        input_size=224,
        color_jitter=.3,
        aa='rand-m9-mstd0.5-inc1',
        train_interpolation='bicubic',
        reprob=.25,
        remode='pixel',
        recount=1,
        eval_crop_ratio=0.875,
        batch_size=128,
        num_workers=16):
    class Args:
        pass
    args = Args()
    args.data_set = data_set
    args.data_path = datapath
    args.input_size = input_size
    args.color_jitter = color_jitter
    args.aa = aa
    args.train_interpolation = train_interpolation
    args.reprob = reprob
    args.remode = remode
    args.recount = recount
    args.eval_crop_ratio = eval_crop_ratio
    train_dataset, _ = build_dataset(is_train=True, args=args)
    val_dataset, _ = build_dataset(is_train=False, args=args)

    train_dataset = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=True
    )
    val_dataset = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=True
    )

    return train_dataset, val_dataset, val_dataset
