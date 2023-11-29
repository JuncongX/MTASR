import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
from models.MTASR import Task
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

        x_32 = self.conv_1(x_16)

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
        # self.classifier = nn.Linear(512, classes_num)
        self.classifier = nn.Sequential(
            nn.Linear(512, setting.classifier_hidden_size),
            # nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(setting.classifier_hidden_size, classes_num),
        )

        self.hr_task = Task()
        self.hr_estimator = nn.Linear(256, 1)
        # self.hr_estimator = nn.Sequential(
        #     nn.Linear(256, setting.classifier_hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(setting.classifier_hidden_size, 1),
        # )

    def forward(self, x, net_type="both"):
        x_16, x_32, x_64, x_128 = self.bvp(x)

        if net_type == "classify":
            state_out = self.state_task(x_16, x_32, x_64, x_128)
            out = self.classifier(F.adaptive_avg_pool1d(state_out, 1).squeeze(-1))
        elif net_type == "hr":
            hr_out = self.hr_task(x_16, x_32, x_64, x_128)
            hr = self.hr_estimator(F.adaptive_avg_pool1d(hr_out, 1).squeeze(-1))
        else:
            state_out = self.state_task(x_16, x_32, x_64, x_128)

            hr_out = self.hr_task(x_16, x_32, x_64, x_128)

            # out = self.classifier(F.adaptive_avg_pool1d(state_out + hr_out, 1).squeeze(-1))
            out = self.classifier(F.adaptive_avg_pool1d(torch.cat((state_out, hr_out), 1), 1).squeeze(-1))
            # out = self.classifier(F.adaptive_avg_pool1d(self.mmtm(state_out, hr_out), 1).squeeze(-1))

            hr = self.hr_estimator(F.adaptive_avg_pool1d(hr_out, 1).squeeze(-1))

        if net_type == "classify":
            return out
        elif net_type == "hr":
            return hr
        else:
            return out, hr

    def init_hidden(self, bs, device):
        h0 = Variable(torch.zeros(self.num_layers, bs, self.hidden_size).to(device))
        c0 = Variable(torch.zeros(self.num_layers, bs, self.hidden_size).to(device))
        return h0, c0


if __name__ == '__main__':
    net = MetaStress()
    in_p = torch.randn((5, 1, 2100))
    out, hr = net(in_p)
    print(out.shape)
    print(hr.shape)

    #
    # def get_parameter_number(net):
    #     total_num = sum(p.numel() for p in net.parameters())
    #     trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #     return {'Total': total_num, 'Trainable': trainable_num}
    #
    #
    # print(get_parameter_number(net))
    #
    # in_p = torch.randn((5, 32, 420))
    # Jnet = JAM(32)
    # out = Jnet(in_p)
    # print(out.shape)
