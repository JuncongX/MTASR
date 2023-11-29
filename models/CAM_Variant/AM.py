import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
from models.MTASR import MetaStress


class AM(nn.Module):
    def __init__(self):
        super(AM, self).__init__()

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.atten = nn.Conv1d(1, 1, 3, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, Y):
        Y_gap = self.gap(Y)
        atten_gap = self.atten(Y_gap.transpose(-1, -2)).transpose(-1, -2)
        atten = self.sigmoid(atten_gap)
        return atten * Y + Y


class Task(nn.Module):
    def __init__(self):
        super(Task, self).__init__()
        self.block_1 = nn.Sequential(
            # TLAM(2100),
            AM(),
            nn.Conv1d(16, 32, 1, 1),
            # nn.BatchNorm1d(32),
            # nn.LeakyReLU(inplace=True)
        )
        self.block_2 = nn.Sequential(
            # TLAM(2100),
            AM(),
            nn.Conv1d(32, 64, 1, 1),
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5, 5)
        )

        self.block_3 = nn.Sequential(
            # TLAM(420),
            AM(),
            nn.Conv1d(64, 128, 1, 1),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5, 5)
        )

        self.block_4 = nn.Sequential(
            AM(),
            nn.Conv1d(128, 256, 1, 1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(2, 2)
        )

    def forward(self, x_16, x_32, x_64, x_128):
        # x_32_a = self.block_1(x_16) + x_32
        x_64_a = self.block_2(x_32) + x_64
        x_128_a = self.block_3(x_64_a) + x_128
        x_256_a = self.block_4(x_128_a)
        return x_256_a


if __name__ == '__main__':
    net = MetaStress()
    net.state_task = Task()
    net.hr_task = Task()
    in_p = torch.randn((5, 1, 2100))
    out, hr, peak = net(in_p)
    print(out.shape)
    print(hr.shape)
    print(peak.shape)

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    print(get_parameter_number(net))
