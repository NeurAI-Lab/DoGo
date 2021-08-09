from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100, STL10
import os
from util.torchlist import ImageFilelist
from models import LinearEvaluation
from util import log, save_checkpoint, tiny_imagenet, CIFAR10Imbalanced
from augmentations import TestTransform
from optimizers.lars import LARC
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_domain_net(args):
    """ DomainNet datasets - QuickDraw, Sketch, ClipArt """
    train_kv = args.eval.dataset.name + "_train.txt"
    test_kv = args.eval.dataset.name + "_test.txt"
    train_dataset = ImageFilelist(root=args.eval.dataset.data_dir,
                                  flist=os.path.join(args.eval.dataset.data_dir, train_kv), transform=TestTransform(args.eval.dataset.img_size))
    test_dataset = ImageFilelist(root=args.eval.dataset.data_dir,
                                 flist=os.path.join(args.eval.dataset.data_dir, test_kv), transform=TestTransform(args.eval.dataset.img_size))
    return train_dataset, test_dataset


def testloaderSimCLR(args, dataset, transform, batchsize, data_dir, val_split=0.15):
    """
    Load test datasets
    """
    if dataset == 'CIFAR100':
        train_d = CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_d = CIFAR100(data_dir, train=False, download=True, transform=transform)
    elif dataset == 'CIFAR10':
        train_d = CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_d = CIFAR10(data_dir, train=False, download=True, transform=transform)
    elif dataset == 'CIFAR10Imbalanced':
        train_d = CIFAR10Imbalanced(root=args.train.dataset.data_dir, train=True, download=True, transform=transform)
        test_d = CIFAR10(data_dir, train=False, download=True, transform=transform)
    elif dataset == 'TinyImageNet':
        train_d, test_d = tiny_imagenet(data_dir, train=False)
    elif dataset == 'STL10':
        train_d = STL10(data_dir, split='train', download=True, transform=transform)
        test_d = STL10(data_dir, split='test', download=True, transform=transform)
    elif dataset == 'quickdraw' or dataset == 'sketch' or dataset == 'clipart':
        train_d, test_d = get_domain_net(args)

    # train - validation split
    val_size = int(val_split * len(train_d))
    train_size = len(train_d) - val_size
    train_d, val_d = random_split(train_d, [train_size, val_size])

    train_loader = DataLoader(train_d, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=args.eval.num_workers)
    val_loader = DataLoader(val_d, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=args.eval.num_workers)
    test_loader = DataLoader(test_d, batch_size=batchsize, shuffle=False, drop_last=True, num_workers=args.eval.num_workers)
    log("Took {} time to load data!".format(datetime.now() - args.start_time))
    return train_loader, val_loader, test_loader


def train_or_val(args, loader, simclr, model, criterion, optimizer=None, scheduler=None, train=False):
    """
    Train Linear model
    """
    loss_epoch = 0
    accuracy_epoch = 0
    simclr.eval()
    if train:
        model.train()
    else:
        model.eval()
        model.zero_grad()

    for step, (x, y) in enumerate(loader):
        x = x.to(args.device)
        y = y.to(args.device)

        x = simclr.f(x)
        feature = torch.flatten(x, start_dim=1)
        output = model(feature)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        loss_epoch += loss.item()
    return loss_epoch, accuracy_epoch


def testSSL(args, writer, simclr):
    for param in simclr.parameters():
        param.requires_grad = False

    linear_model = LinearEvaluation(simclr.projection_size, args.eval.dataset.classes)
    if torch.cuda.device_count() > 1:
        linear_model = nn.DataParallel(linear_model)
    linear_model = linear_model.to(args.device)

    scheduler = None
    if args.eval.optimizer.name == 'adam':
        optimizer = optim.Adam(linear_model.parameters(), lr=args.eval.optimizer.lr, weight_decay=args.eval.optimizer.weight_decay)
    elif args.eval.optimizer.name == 'SGD':
        optimizer = optim.SGD(linear_model.parameters(), lr=args.eval.optimizer.lr,
                               weight_decay=args.eval.optimizer.weight_decay, momentum=args.eval.optimizer.momentum)

    if args.eval.optimizer.scheduler:
            scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=3e-4)

    transform = TestTransform(args.eval.dataset.img_size)
    train_loader, val_loader, test_loader = testloaderSimCLR(args, args.eval.dataset.name, transform,
                                                             args.eval.batchsize, args.eval.dataset.data_dir)
    loss_criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    log('Testing SSL Model on {}.................'.format(args.eval.dataset.name))
    _, ck_name = os.path.split(args.eval.model_path)
    for epoch in range(1, args.eval.epochs + 1):
        loss_epoch, accuracy_epoch = train_or_val(args, train_loader, simclr, linear_model, loss_criterion, optimizer, scheduler, train=True)
        log(f"Epoch [{epoch}/{args.eval.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}")

        loss_epoch1, accuracy_epoch1 = train_or_val(args, val_loader, simclr, linear_model, loss_criterion, train=False)
        val_accuracy = accuracy_epoch1 / len(test_loader)
        log(f"Epoch [{epoch}/{args.eval.epochs}] \t Validation accuracy {val_accuracy}")

        if best_acc < val_accuracy:
            best_acc = val_accuracy
            log('Best accuracy achieved so far: {}'.format(best_acc))
            if torch.cuda.device_count() > 1:
                # Save DDP model's module
                save_checkpoint(state_dict=linear_model.module.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_best_linear_model_{}_{}.pth'.format(args.eval.dataset.name, ck_name))
            else:
                save_checkpoint(state_dict=linear_model.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_best_linear_model_{}_{}.pth'.format(args.eval.dataset.name, ck_name))
        writer.add_scalar("Accuracy/train{}".format(args.eval.dataset.name), accuracy_epoch / len(train_loader), epoch)
        writer.add_scalar("Accuracy/val{}".format(args.eval.dataset.name), accuracy_epoch1 / len(test_loader), epoch)

    # Load best linear model and run inference on test set
    state_dict = torch.load(os.path.join(args.log_dir, 'checkpoint_best_linear_model_{}_{}.pth'.format(args.eval.dataset.name, ck_name)), map_location=args.device)
    linear_best_model = LinearEvaluation(simclr.projection_size, args.eval.dataset.classes)
    linear_best_model.load_state_dict(state_dict)
    linear_best_model = linear_best_model.cuda()
    test_loss, test_acc = train_or_val(args, test_loader, simclr, linear_best_model, loss_criterion, train=False)
    test_acc = test_acc / len(test_loader)
    log(f" Test accuracy : {test_acc}")
    writer.add_text("Test Accuracy {} :".format(args.eval.dataset.name), "{}".format(test_acc))


def test_all_datasets(args, writer, model):
    """
    Test all datasets for linear evaluation
    """
    if args.eval.dataset.name is None or args.eval.dataset.data_dir is None:
        # CIFAR10
        # args.eval.dataset.img_size = 32
        # args.eval.dataset.name = "CIFAR10"
        # args.eval.dataset.data_dir = "/data/input/datasets/CIFAR-10"
        # args.eval.dataset.classes = 10
        # testSSL(args, writer, model)
        # # # CIFAR100
        # args.eval.dataset.img_size = 32
        # args.eval.dataset.name = "CIFAR100"
        # args.eval.dataset.data_dir = "/input/CIFAR-100"
        # args.eval.dataset.classes = 100
        # testSSL(args, writer, model)
        # # STL10
        # args.eval.dataset.img_size = 96
        # args.eval.dataset.name = "STL10"
        # args.eval.dataset.data_dir = "/input/STL-10"
        # args.eval.dataset.classes = 10
        # testSSL(args, writer, model)
        # # TinyImageNet
        # args.eval.dataset.img_size = 64
        # args.eval.dataset.name = "TinyImageNet"
        # args.eval.dataset.data_dir = "/input/tiny_imagenet/tiny-imagenet-200"
        # args.eval.dataset.classes = 200
        # testSSL(args, writer, model)
        # # quickdraw
        # args.eval.dataset.img_size = 64
        # args.eval.dataset.name = "quickdraw"
        # args.eval.dataset.data_dir = "/input/DomainNet"
        # args.eval.dataset.classes = 345
        # testSSL(args, writer, model)
        # # clipart
        # args.eval.dataset.img_size = 64
        # args.eval.dataset.name = "clipart"
        # args.eval.dataset.data_dir = "/input/DomainNet"
        # args.eval.dataset.classes = 345
        # testSSL(args, writer, model)
        # # sketch
        # args.eval.dataset.img_size = 64
        # args.eval.dataset.name = "sketch"
        # args.eval.dataset.data_dir = "/input/DomainNet"
        # args.eval.dataset.classes = 345
        # testSSL(args, writer, model)
        print()
    else:
        testSSL(args, writer, model)
