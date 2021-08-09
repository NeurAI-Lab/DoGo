"""
SimSiam: Exploring Simple Siamese Representation Learning
Code from their pseudocode
"""
import torch.nn as nn


class SimSiamLoss(nn.Module):

    def __init__(self):
        super(SimSiamLoss, self).__init__()

    def forward(self, zx, zy, px, py):
        loss = -(zx.detach() * py).sum(dim=1).mean()
        loss += -(zy.detach() * px).sum(dim=1).mean()
        return loss / 2