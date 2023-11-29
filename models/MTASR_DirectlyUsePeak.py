import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
from models.MTASR import Task, MixA_Module
import models.setting as setting


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
        # self.peak_main = nn.Sequential(
        #     nn.Conv1d(16, 8, 3, 1, 1),
        #     nn.BatchNorm1d(8),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(8, 4, 3, 1, 1),
        #     nn.BatchNorm1d(4),
        #     nn.LeakyReLU(inplace=True)
        # )
        #
        # self.peak_residual = nn.Sequential(
        #     nn.Conv1d(16, 4, 3, 1, 1),
        #     nn.BatchNorm1d(4),
        #     nn.LeakyReLU(inplace=True)
        # )
        #
        # self.peak_out = nn.Sequential(
        #     nn.Conv1d(4, 1, 3, 1, 1),
        #     nn.Sigmoid()
        # )

        self.MixA_Module = MixA_Module()

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

    def forward(self, x, peak):
        x_16 = self.conv_peak(x)  # B, 16, 350

        x_16_SA, attention_16 = self.MixA_Module(x_16, peak.unsqueeze(1))  # B, 16, 350

        x_32 = self.conv_1(x_16_SA)

        x_64 = self.conv_2(F.max_pool1d(x_32, 5, 5))

        x_128 = self.conv_3(F.max_pool1d(x_64, 5, 5))

        return x_16, x_32, x_64, x_128


class MetaStress(nn.Module):
    def __init__(self, classes_num=2, hidden_size=128, num_layers=2):
        super(MetaStress, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bvp = BVPFeatrueExtraction()

        self.state_task = Task()

        self.classifier = nn.Sequential(
            nn.Linear(512, setting.classifier_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(setting.classifier_hidden_size, classes_num),
        )

        self.hr_task = Task()
        self.hr_estimator = nn.Linear(256, 1)

    def forward(self, x, peak, net_type="both"):
        x_16, x_32, x_64, x_128 = self.bvp(x, peak)

        if net_type == "classify":
            state_out = self.state_task(x_16, x_32, x_64, x_128)
            out = self.classifier(F.adaptive_avg_pool1d(state_out, 1).squeeze(-1))
        elif net_type == "hr":
            hr_out = self.hr_task(x_16, x_32, x_64, x_128)
            hr = self.hr_estimator(F.adaptive_avg_pool1d(hr_out, 1).squeeze(-1))
        else:
            state_out = self.state_task(x_16, x_32, x_64, x_128)

            hr_out = self.hr_task(x_16, x_32, x_64, x_128)

            out = self.classifier(F.adaptive_avg_pool1d(torch.cat((state_out, hr_out), 1), 1).squeeze(-1))

            hr = self.hr_estimator(F.adaptive_avg_pool1d(hr_out, 1).squeeze(-1))

        if net_type == "classify":
            return out
        elif net_type == "hr":
            return hr
        else:
            return out, hr


if __name__ == '__main__':
    net = MetaStress()
    in_p = torch.randn((5, 1, 2100))
    peak = torch.randn((5, 2100))
    out, hr = net(in_p, peak)
    print(out.shape)
    print(hr.shape)


    # def get_parameter_number(net):
    #     total_num = sum(p.numel() for p in net.parameters())
    #     trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #     return {'Total': total_num, 'Trainable': trainable_num}
    #
    #
    # print(get_parameter_number(net))
    #
    # ecam = ECAM(128)
    # print(get_parameter_number(ecam))
    #
    # clam = CLAM(64)
    # print(get_parameter_number(clam))
