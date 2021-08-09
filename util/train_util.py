import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import os

from util.utils import save_checkpoint, log
from util.test import test_all_datasets
from criterion import NTXent


def get_criteria(args):
    """
    Loss criterion / criteria selection for training
    """
    criteria = {
        'ntxent': [NTXent(args), args.train.criterion_weight[0]]
    }

    return criteria


def write_scalar(writer, total_loss, loss_p_c, leng, epoch):
    """
    Add Loss scalars to tensorboard
    """
    writer.add_scalar("Total_Loss/train", total_loss / leng, epoch)
    for k in loss_p_c:
        writer.add_scalar("{}_Loss/train".format(k), loss_p_c[k] / leng, epoch)


def train_one_epoch(args, train_loader, model, criteria, optimizer, scheduler, epoch):
    """
    Train one epoch of SSL model
    """
    loss_per_criterion = {}
    total_loss = 0
    for i, ((x1, y1), targets) in enumerate(train_loader):
        x1 = x1.cuda(device=args.device)
        y1 = y1.cuda(device=args.device)
        optimizer.zero_grad()

        if args.train.model == 'simclr':
            _, _, zx, zy = model(x1, y1)
        elif args.train.model == 'simsiam':
            fx, fy, zx, zy, px, py = model(x1, y1)

        # Multiple loss aggregation
        loss = torch.tensor(0).to(args.device)
        for k in criteria:
            if k == 'ntxent':
                criterion_loss = criteria[k][0](zx, zy)
            elif k == 'simsiam':
                criterion_loss = criteria[k][0](zx, zy, px, py)
            if k not in loss_per_criterion:
                loss_per_criterion[k] = criterion_loss
            else:
                loss_per_criterion[k] += criterion_loss
            loss = torch.add(loss, torch.mul(criterion_loss, criteria[k][1]))

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if i % 50 == 0:
            log("Batch {}/{}. Loss: {}.  Time elapsed: {} ".format(i, len(train_loader), loss.item(),
                                                                   datetime.now() - args.start_time))
        total_loss += loss.item()
    return total_loss, loss_per_criterion


def trainSSL(args, model, train_loader, optimizer, criteria, writer, scheduler=None):
    """
    Train a SSL model
    """
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log('Model converted to DP model with {} cuda devices'.format(torch.cuda.device_count()))
    model = model.to(args.device)

    for epoch in tqdm(range(1, args.train.epochs + 1)):
        model.train()
        total_loss, loss_per_criterion = train_one_epoch(args, train_loader, model, criteria, optimizer, scheduler, epoch)
        log("Epoch {}/{}. Total Loss: {}.   Time elapsed: {} ".
            format(epoch, args.train.epochs, total_loss / len(train_loader), datetime.now() - args.start_time))

        write_scalar(writer, total_loss, loss_per_criterion, len(train_loader), epoch)

        # Save the model at specific checkpoints
        if epoch % 10 == 0:
            if torch.cuda.device_count() > 1:
                save_checkpoint(state_dict=model.module.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model_{}.pth'.format(epoch))
            else:
                save_checkpoint(state_dict=model.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model_{}.pth'.format(epoch))

    log("Total training time {}".format(datetime.now() - args.start_time))

    # Test the SSl Model
    if torch.cuda.device_count() > 1:
        test_all_datasets(args, writer, model.module)
    else:
        test_all_datasets(args, writer, model)

    writer.close()
