import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):
    """
    KL-Divergence symmetric loss between two distributions
    Used in here for knowledge distillation
    """

    def __init__(self):
        super(KLLoss, self).__init__()
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, zxs, zys, zxt, zyt, temperature=0.1):
        sim_s = self.similarity_f(zxs.unsqueeze(1), zys.unsqueeze(0)) / temperature
        sim_s = F.softmax(sim_s, dim=1)
        sim_t = self.similarity_f(zxt.unsqueeze(1), zyt.unsqueeze(0)) / temperature
        sim_t = F.softmax(sim_t, dim=1)
        loss_s = F.kl_div(sim_s.log(), sim_t.detach())
        loss_t = F.kl_div(sim_t.log(), sim_s.detach())
        return loss_s, loss_t