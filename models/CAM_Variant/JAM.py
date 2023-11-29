import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
from models.MTASR import MetaStress, CLAM


# Understanding and Learning Discriminant Features based on Multiattention 1DCNN for Wheelset Bearing Fault Diagnosis
class JAM(nn.Module):
    def __init__(self, in_w, in_c):
        super(JAM, self).__init__()
        self.eam_conv1x1 = nn.Conv1d(in_w, 1, 1, 1, bias=False)
        self.eam_conv3x3 = nn.Conv1d(in_w, in_w, 3, 1, 1, bias=False)
        self.clam = CLAM(in_c)

    def forward(self, Y):
        Y_T = Y.transpose(1, 2)
        F_ecm_1 = torch.sigmoid(self.eam_conv1x1(Y_T))
        F_ecm_2 = F.leaky_relu(self.eam_conv3x3(Y_T))
        F_ecm = F_ecm_1 * F_ecm_2
        ECM_out = F_ecm.transpose(1, 2) + Y
        return self.clam(ECM_out)


class Task(nn.Module):
    def __init__(self):
        super(Task, self).__init__()
        self.block_1 = nn.Sequential(
            # TLAM(2100),
            JAM(2100, 16),
            nn.Conv1d(16, 32, 1, 1),
            # nn.BatchNorm1d(32),
            # nn.LeakyReLU(inplace=True)
        )
        self.block_2 = nn.Sequential(
            # TLAM(2100),
            JAM(2100, 32),
            nn.Conv1d(32, 64, 1, 1),
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5, 5)
        )

        self.block_3 = nn.Sequential(
            # TLAM(420),
            JAM(420, 64),
            nn.Conv1d(64, 128, 1, 1),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5, 5)
        )

        self.block_4 = nn.Sequential(
            JAM(84, 128),
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
    #
    # ecam = ECAM(128)
    # print(get_parameter_number(ecam))
    #
    # clam = CLAM(64)
    # print(get_parameter_number(clam))
