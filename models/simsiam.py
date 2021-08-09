import torch.nn as nn
import torch
import torch.nn.functional as F
from models.helper import get_encoder


class SimSiam(nn.Module):
    """
    Exploring Simple Siamese Representation Learning
    https://arxiv.org/abs/2011.10566
    """
    def __init__(self, args, img_size, backbone='resnet50'):
        super(SimSiam, self).__init__()
        self.f, args.projection_size = get_encoder(backbone, img_size)
        if img_size >= 100:
            args.projection_size = self.f.fc.out_features

        # projection MLP
        self.g = nn.Sequential(
            nn.Linear(args.projection_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            # nn.Linear(2048, 2048),
            # nn.BatchNorm1d(2048),
            # nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
        )

        # predictor MLP
        self.h = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, args.train.n_proj),
        )

    def forward(self, x, y=None):
        x = self.f(x)
        fx = torch.flatten(x, start_dim=1)
        zx = self.g(fx)
        px = self.h(zx)

        if y is not None:
            y = self.f(y)
            fy = torch.flatten(y, start_dim=1)
            zy = self.g(fy)
            py = self.h(zy)
            return F.normalize(fx, dim=1), F.normalize(fy, dim=1), \
                   F.normalize(zx, dim=1), F.normalize(zy, dim=1), \
                   F.normalize(px, dim=1), F.normalize(py, dim=1)
        else:
            return F.normalize(fx, dim=-1), F.normalize(zx, dim=-1)


