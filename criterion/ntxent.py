import torch
import torch.nn as nn
from util.utils import positive_mask


class NTXent(nn.Module):
    """
    The Normalized Temperature-scaled Cross Entropy Loss
    Source: https://github.com/Spijkervet/SimCLR
    """

    def __init__(self, args):
        super(NTXent, self).__init__()
        self.batch_size = args.train.batchsize
        self.temperature = args.train.temperature
        self.device = args.device
        self.mask = positive_mask(args.train.batchsize)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.N = 2 * self.batch_size

    def forward(self, zx, zy):
        """
        zx: projection output of batch zx
        zy: projection output of batch zy
        :return: normalized loss
        """
        positive_samples, negative_samples = self.sample_no_dict(zx, zy)
        labels = torch.zeros(self.N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= self.N
        return loss

    def sample_no_dict(self, zx, zy):
        """
        Positive and Negative sampling without dictionary
        """
        z = torch.cat((zx, zy), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # Since projections are already normalized using F.normalize,
        # below function can be used instead of CosineSimilarity
        # sim = torch.div(torch.matmul(z, z.T), self.temperature)

        # Extract positive samples
        sim_xy = torch.diag(sim, self.batch_size)
        sim_yx = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_xy, sim_yx), dim=0).reshape(self.N, 1)

        # Extract negative samples
        negative_samples = sim[self.mask].reshape(self.N, -1)
        return positive_samples, negative_samples

