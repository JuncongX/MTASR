 # An end-to-end atrial fibrillation detection by a novel residual-based temporal attention convolutional neural network with exponential nonlinearity loss
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

        self.conv_a_1 = nn.Sequential(
            nn.Conv1d(16, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
        )

        self.conv_a_2 = nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
        )

        self.conv_a_3 = nn.Sequential(
            nn.Conv1d(16, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
        )

        self.conv_a_4 = nn.Sequential(
            nn.Conv1d(16, 1, 1, 1),
            nn.Sigmoid()
        )

        self.conv_m = nn.Sequential(
            nn.Conv1d(16, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, X):
        x_a_1 = self.conv_a_1(X)
        x_a_down = self.conv_a_2(x_a_1)
        x_a_up = self.conv_a_3(F.interpolate(x_a_down, scale_factor=2))
        x_a_2 = x_a_1 + x_a_up
        x_a_w = self.conv_a_4(x_a_2)

        x_m = self.conv_m(X)
        return x_m * x_a_w + X, x_a_w


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
    # in_p = torch.randn((5, 16, 2100))
    # sa = TA_block()
    # out = sa(in_p)
    # print(out.shape)
