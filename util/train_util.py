import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from util import save_checkpoint, log
from util.test import test_all_datasets
from criterion import NTXent, KLLoss, SimSiamLoss
from models import SimCLR, SimSiam


def get_criteria(args):
    """
    Loss criterion / criteria selection for training
    """
    criteria = {
        'ntxent': [NTXent(args), args.train.criterion_weight[0]],
        'simsiam': [SimSiamLoss(), args.train.criterion_weight[1]],
        'kl': [KLLoss(), args.train.criterion_weight[2]]
    }

    return criteria


def write_scalar(writer, total_loss, loss_p_c, leng, epoch):
    """
    Add Loss scalars to tensorboard
    """
    writer.add_scalar("Total_Loss/train", total_loss / leng, epoch)
    for k in loss_p_c:
        writer.add_scalar("{}_Loss/train".format(k), loss_p_c[k] / leng, epoch)


def insert_loss_per_criterion(loss_p_c, key, loss_per_batch):
    if key not in loss_p_c:
        loss_p_c[key] = loss_per_batch
    else:
        loss_p_c[key] += loss_per_batch


def train_one_epoch(args, train_loader, model1, model2, criteria, optimizers, schedulers, epoch):
    """
    Train one epoch of SSL model
    """
    loss_per_criterion = {}
    total_loss = 0

    for i, (inputs, targets) in enumerate(train_loader):
        x1 = inputs[0][0].cuda(device=args.device)
        y1 = inputs[0][1].cuda(device=args.device)
        x2 = inputs[1][0].cuda(device=args.device)
        y2 = inputs[1][1].cuda(device=args.device)
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss_1 = torch.tensor(0).to(args.device)
        loss_2 = torch.tensor(0).to(args.device)

        if isinstance(model1, SimCLR):
            fx1, fy1, px1, py1 = model1(x1, y1)
            loss_1 = criteria['ntxent'][0](px1, py1)
            insert_loss_per_criterion(loss_per_criterion, 'ntxent_M1', loss_1)
        elif isinstance(model1, SimSiam):
            fx1, fy1, zx1, zy1, px1, py1 = model1(x1, y1)
            loss_1 = criteria['simsiam'][0](zx1, zy1, px1, py1)
            insert_loss_per_criterion(loss_per_criterion, 'simsiam_M1', loss_1)

        if isinstance(model2, SimCLR):
            fx2, fy2, px2, py2 = model2(x2, y2)
            loss_2 = criteria['ntxent'][0](px2, py2)
            insert_loss_per_criterion(loss_per_criterion, 'ntxent_M2', loss_2)
        elif isinstance(model2, SimSiam):
            fx2, fy2, zx2, zy2, px2, py2 = model2(x2, y2)
            loss_2 = criteria['simsiam'][0](zx2, zy2, px2, py2)
            insert_loss_per_criterion(loss_per_criterion, 'simsiam_M2', loss_2)

        # KL Divergence
        if isinstance(model1, SimSiam) and isinstance(model2, SimSiam):
            loss_1_KL, _ = criteria['kl'][0](px1, py1, zx2, zy2, temperature=args.train.KL_temperature)
            _, loss_2_KL = criteria['kl'][0](zx1, zy1, px2, py2, temperature=args.train.KL_temperature)
        else:
            loss_1_KL, loss_2_KL = criteria['kl'][0](px1, py1, px2, py2, temperature=args.train.KL_temperature)
        insert_loss_per_criterion(loss_per_criterion, 'kl_M1', loss_1_KL)
        insert_loss_per_criterion(loss_per_criterion, 'kl_M2', loss_2_KL)

        loss_1 = torch.add(loss_1, torch.mul(loss_1_KL, args.train.criterion_weight[2]))
        loss_2 = torch.add(loss_2, torch.mul(loss_2_KL, args.train.criterion_weight[3]))

        loss_1.backward()
        loss_2.backward()

        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            if scheduler is not None:
                scheduler.step()
        if i % 50 == 0:
            log("Batch {}/{}. Student1 Loss: {}. Student2 Loss: {}.  Time elapsed: {} ".format(i, len(train_loader), loss_1.item(), loss_2.item(),
                                                                   datetime.now() - args.start_time))
        total_loss += loss_1.item() + loss_2.item()
    return total_loss, loss_per_criterion


def trainSSL(args, models, train_loader, optimizers, criteria, writer, schedulers=None):
    """
    Train a SSL model
    """
    model1 = models[0]
    model2 = models[1]
    model1.train()
    model2.train()
    if torch.cuda.device_count() > 1:
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)
        log('Model converted to DP model with {} cuda devices'.format(torch.cuda.device_count()))
    model1 = model1.to(args.device)
    model2 = model2.to(args.device)

    for epoch in tqdm(range(1, args.train.epochs + 1)):
        model1.train()
        model2.train()
        total_loss, loss_per_criterion = train_one_epoch(args, train_loader, model1, model2, criteria, optimizers, schedulers, epoch)
        log("Epoch {}/{}. Total Loss: {}.   Time elapsed: {} ".
            format(epoch, args.train.epochs, total_loss / len(train_loader), datetime.now() - args.start_time))

        write_scalar(writer, total_loss, loss_per_criterion, len(train_loader), epoch)

        # Save the model at specific checkpoints
        if epoch % 10 == 0:
            if torch.cuda.device_count() > 1:
                save_checkpoint(state_dict=model1.module.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model1_{}_{}.pth'.format(epoch, args.train.models[0]['backbone']))
                save_checkpoint(state_dict=model2.module.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model2_{}_{}.pth'.format(epoch, args.train.models[1]['backbone']))
            else:
                save_checkpoint(state_dict=model1.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model1_{}_{}.pth'.format(epoch, args.train.models[0]['backbone']))
                save_checkpoint(state_dict=model2.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model2_{}_{}.pth'.format(epoch, args.train.models[1]['backbone']))

    log("Total training time {}".format(datetime.now() - args.start_time))

    writer.close()
