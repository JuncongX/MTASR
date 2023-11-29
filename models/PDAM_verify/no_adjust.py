import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
from models.MTASR import MetaStress


class MixA_Module(nn.Module):
    def __init__(self):
        super(MixA_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, peak):
        # q = self.q_w(F.adaptive_avg_pool1d(x.transpose(1, 2), 1).transpose(1, 2))
        Attention_weight = self.softmax(peak)
        output = x * Attention_weight + x
        return output, Attention_weight


if __name__ == '__main__':
    net = MetaStress()
    net.bvp.MixA_Module = MixA_Module()
    in_p = torch.randn((5, 1, 2100))
    out, hr, peak = net(in_p)
    print(out.shape)
    print(hr.shape)
    print(peak.shape)

    print(net)
