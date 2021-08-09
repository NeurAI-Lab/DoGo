import torch
from time import ctime
import os
from torch.utils.tensorboard import SummaryWriter
import logging
from augmentations.simclr_transform import SimCLRTransform
from util.torchlist import ImageFilelist
from augmentations import TestTransform
import numpy as np
from torchvision.datasets import CIFAR10


def tiny_imagenet(data_root, img_size=64, train=True, transform=None):
    """
    TinyImageNet dataset
    """
    train_kv = "train_kv_list.txt"
    test_kv = "val_kv_list.txt"
    if train:
        train_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, train_kv), transform=transform)
        return train_dataset
    else:
        train_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, train_kv),
                                      transform=TestTransform(img_size))
        test_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, test_kv),
                                     transform=TestTransform(img_size))
        return train_dataset, test_dataset


def positive_mask(batch_size):
    """
    Create a mask for masking positive samples
    :param batch_size:
    :return: A mask that can segregate 2(N-1) negative samples from a batch of N samples
    """
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=torch.bool)
    mask[torch.eye(N).byte()] = 0
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


def summary_writer(args, log_dir=None, filename_suffix=''):
    """
    Create a tensorboard SummaryWriter
    """
    if log_dir is None:
        args.log_dir = os.path.join(args.train.save_dir, "_bs_{}".format(args.train.batchsize),
                                    ctime().replace(' ', '_'))
        mkdir(args.log_dir)
    else:
        args.log_dir = log_dir
    writer = SummaryWriter(log_dir=args.log_dir, filename_suffix=filename_suffix)
    print("logdir = {}".format(args.log_dir))
    return writer


def mkdir(path):
    """
    Creates new directory if not exists
    @param path:  folder path
    """
    if not os.path.exists(path):
        print("creating {}".format(path))
        os.makedirs(path, exist_ok=True)


def logger(args, filename=None):
    """
    Creates a basic config of logging
    @param args: Namespace instance with parsed arguments
    @param filename: None by default
    """
    if filename is None:
        filename = os.path.join(args.log_dir, 'train.log')
    else:
        filename = os.path.join(args.log_dir, filename)
    logging.basicConfig(filename=filename, level=logging.DEBUG, format='%(message)s')
    print("logfile created")


def log(msg):
    """
    print and log console messages
    @param msg: string message
    """
    print(msg)
    logging.debug(msg)


def save_checkpoint(state_dict, args, epoch, filename=None):
    """
    @param state_dict: model state dictionary
    @param args: system arguments
    @param epoch: epoch
    @param filename: filename for saving the checkpoint. Do not include whole path as path is appended in the code
    """
    if filename is None:
        path = os.path.join(args.log_dir + "/" + "checkpoint_{}.pth".format(epoch))
    else:
        path = os.path.join(args.log_dir + "/" + filename)

    torch.save(state_dict, path)
    log("checkpoint saved at {} after {} epochs".format(path, epoch))
    return path


class CIFAR10Imbalanced(CIFAR10):
    """@author Fahad Sarfaraz

  CIFAR10 dataset, with support for randomly corrupt labels.

  Params
  ------
  num_classes: int
    Default 10. The number of classes in the dataset.
  """

    def __init__(self, gamma=0.2, n_min=250, n_max=5000, num_classes=10, **kwargs):
        super(CIFAR10Imbalanced, self).__init__(**kwargs)
        log("\n The gamma value for imbalanced CIFAR10: {} \n".format(gamma))
        self.num_classes = num_classes
        self.gamma = gamma
        self.n_min = n_min
        self.n_max = n_max
        self.imbalanced_dataset()

    def imbalanced_dataset(self):
        X = np.array([[1, -self.n_max], [1, -self.n_min]])
        Y = np.array([self.n_max, self.n_min * 10 ** (self.gamma)])
        a, b = np.linalg.solve(X, Y)
        classes = list(range(1, self.num_classes + 1))
        imbal_class_counts = []
        for c in classes:
            num_c = int(np.round(a / (b + (c) ** (self.gamma))))
            print(c, num_c)
            imbal_class_counts.append(num_c)

        targets = np.array(self.targets)
        # Get class indices
        class_indices = [np.where(targets == i)[0] for i in range(self.num_classes)]
        # Get imbalanced number of instances
        imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in
                               zip(class_indices, imbal_class_counts)]
        imbal_class_indices = np.hstack(imbal_class_indices)

        # Set target and data to dataset
        self.targets = targets[imbal_class_indices]
        self.data = self.data[imbal_class_indices]
        print(len(self.targets))
        assert len(self.targets) == len(self.data)
