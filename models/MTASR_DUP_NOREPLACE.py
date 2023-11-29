import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import models.setting as setting


class MixA_Module(nn.Module):
    def __init__(self):
        super(MixA_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        # self.q_w = nn.Conv1d(1, 1, 3, 1, 1)
        self.v_w = nn.Conv1d(16, 1, 3, 1, 1)

    def forward(self, x, attn, peak):
        # q = self.q_w(F.adaptive_avg_pool1d(x.transpose(1, 2), 1).transpose(1, 2))
        v = self.v_w(x)
        Attention_weight = self.softmax(torch.sigmoid(v) + attn + peak)
        output = x * Attention_weight + x
        return output, Attention_weight


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
        self.peak_main = nn.Sequential(
            nn.Conv1d(16, 8, 3, 1, 1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 4, 3, 1, 1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(inplace=True)
        )

        self.peak_residual = nn.Sequential(
            nn.Conv1d(16, 4, 3, 1, 1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(inplace=True)
        )

        self.peak_out = nn.Sequential(
            nn.Conv1d(4, 1, 3, 1, 1),
            nn.Sigmoid()
        )

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
        x_peak_main = self.peak_main(x_16)  # B, 4, 350
        x_peak_residual = self.peak_residual(x_16)  # B, 4, 350
        x_peak_p = x_peak_main + x_peak_residual
        x_peak_out = self.peak_out(x_peak_p)
        x_peak = x_peak_out[:, 0, :]  # B, 210

        x_16_SA, attention_16 = self.MixA_Module(x_16, x_peak_out, peak.unsqueeze(1))  # B, 16, 350

        x_32 = self.conv_1(x_16_SA)

        x_64 = self.conv_2(F.max_pool1d(x_32, 5, 5))

        x_128 = self.conv_3(F.max_pool1d(x_64, 5, 5))

        return x_16, x_32, x_64, x_128, x_peak


class ECAM(nn.Module):
    def __init__(self, in_channel, b=1, gama=2):
        super(ECAM, self).__init__()

        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size + 1
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.atten = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

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
            ECAM(16),
            nn.Conv1d(16, 32, 1, 1),
            # nn.BatchNorm1d(32),
            # nn.LeakyReLU(inplace=True)
        )
        self.block_2 = nn.Sequential(
            # TLAM(2100),
            ECAM(32),
            nn.Conv1d(32, 64, 1, 1),
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5, 5)
        )

        self.block_3 = nn.Sequential(
            # TLAM(420),
            ECAM(64),
            nn.Conv1d(64, 128, 1, 1),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5, 5)
        )

        self.block_4 = nn.Sequential(
            ECAM(128),
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
        #     nn.Linear(256, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 1),
        # )

    def forward(self, x, peak_x, net_type="both"):
        x_16, x_32, x_64, x_128, _ = self.bvp(x, peak_x)

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
    peak = torch.randn((5, 2100))
    out, hr = net(in_p, peak)
    print(out.shape)
    print(hr.shape)
    # print(peak.shape)


    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    print(get_parameter_number(net))
    #
    # in_p = torch.randn((5, 32, 420))
    # Jnet = JAM(32)
    # out = Jnet(in_p)
    # print(out.shape)
