# ECG-based multi-class arrhythmia detection using spatio-temporal attention-based convolutional recurrent neural network

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

from models.MTASR import MetaStress


class TA_block(nn.Module):
    def __init__(self):
        super(TA_block, self).__init__()
        self.conv = nn.Conv1d(2, 1, 7, 1, 3)

    def forward(self, x):
        max_pooling = F.adaptive_max_pool1d(x.transpose(2, 1), 1)
        avg_pooling = F.adaptive_avg_pool1d(x.transpose(2, 1), 1)
        concat_pooling = torch.cat((max_pooling.transpose(2, 1), avg_pooling.transpose(2, 1)), 1)
        out = torch.sigmoid(self.conv(concat_pooling))
        return out * x + x, out


class BVPFeatrueExtraction(nn.Module):

    def __init__(self):
        super(BVPFeatrueExtraction, self).__init__()
        self.conv_peak = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            # nn.Conv1d(16, 16, 3, 1, 1),
            # nn.BatchNorm1d(16),
            # nn.LeakyReLU(inplace=True),
        )
        self.TA = TA_block()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(16, 32, 9, 1, 4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 32, 9, 1, 4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True)

        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(32, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x_16 = self.conv_peak(x)  # B, 16, 350

        x_16_SA, x_peak = self.TA(x_16)

        x_32 = self.conv_1(x_16_SA)

        x_64 = self.conv_2(F.max_pool1d(x_32, 5, 5))

        x_128 = self.conv_3(F.max_pool1d(x_64, 5, 5))

        return x_16, x_32, x_64, x_128, x_peak


if __name__ == '__main__':
    net = MetaStress()
    net.bvp = BVPFeatrueExtraction()
    in_p = torch.randn((5, 1, 2100))
    out, hr, peak = net(in_p)
    print(out.shape)
    print(hr.shape)
    print(peak.shape)

    print(net)
