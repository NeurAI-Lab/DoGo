"""
SSL training script script.
Use config/*.yaml by changing it appropriately
"""

import torch
import os
import numpy as np
import random
from datetime import datetime
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.insert(0, '.')
from util.utils import logger, summary_writer, log, tiny_imagenet, CIFAR10Imbalanced
from util.train_util import trainSSL, get_criteria
from augmentations import SimCLRTransform, SimSiamTransform
from config.option import Options
from models import SimCLR, SimSiam
np.random.seed(20)
random.seed(20)
torch.manual_seed(20)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def trainloaderSSL(args, transform, imagenet_split='train'):
    """
    Load training data through DataLoader
    """
    if args.train.dataset.name == 'CIFAR100':
        train_dataset = CIFAR100(args.train.dataset.data_dir, train=True, download=True, transform=transform)
    elif args.train.dataset.name == 'ImageNet':
        train_dataset = ImageFolder(os.path.join(args.train.dataset.data_dir, imagenet_split), transform=transform)
    elif args.train.dataset.name == 'CIFAR10':
        train_dataset = CIFAR10(args.train.dataset.data_dir, train=True, download=True, transform=transform)
    elif args.train.dataset.name == 'CIFAR10Imbalanced':
        train_dataset = CIFAR10Imbalanced(root=args.train.dataset.data_dir, train=True, download=True, transform=transform)
    elif args.train.dataset.name == 'TinyImageNet':
        train_dataset = tiny_imagenet(args.train.dataset.data_dir, train=True, transform=transform)
    elif args.train.dataset.name == 'STL10':
        train_dataset = STL10(args.train.dataset.data_dir, split="unlabeled", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.train.batchsize, shuffle=True, drop_last=True, num_workers=args.train.num_workers)
    log("Took {} time to load data!".format(datetime.now() - args.start_time))
    return train_loader


if __name__ == "__main__":
    args = Options().parse()
    args.writer = summary_writer(args)
    logger(args)
    args.start_time = datetime.now()
    log("Starting at  {}".format(datetime.now()))
    log("arguments parsed: {}".format(args))
    criterion = get_criteria(args)

    if args.train.model == 'simclr':
        model = SimCLR(args, args.train.dataset.img_size, backbone=args.train.backbone)
        transform = SimCLRTransform(args.train.dataset.img_size)
        train_loader = trainloaderSSL(args, transform)
    elif args.train.model == 'simsiam':
        model = SimSiam(args, args.train.dataset.img_size, backbone=args.train.backbone)
        transform = SimSiamTransform(args.train.dataset.img_size)
        train_loader = trainloaderSSL(args, transform)

    scheduler = None
    if args.train.optimizer.name == 'adam':
        optimizer = Adam(model.parameters(), lr=args.train.optimizer.lr, weight_decay=args.train.optimizer.weight_decay)
    if args.train.optimizer.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.train.epochs, eta_min=3e-4)
    trainSSL(args, model, train_loader, optimizer, criterion, args.writer, scheduler)



