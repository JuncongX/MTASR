import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CLAM(nn.Module):
    def __init__(self, input_channels):
        super(CLAM, self).__init__()
        self.atten = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(input_channels, input_channels // 2, 1, 1),
            nn.ReLU(),
            nn.Conv1d(input_channels // 2, input_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, Y):
        atten = self.atten(Y)
        return atten * Y + Y


class FS_Network(nn.Module):

    def __init__(self):
        super(FS_Network, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 16, 9, 1, 4),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, 9, 1, 4),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(16, 32, 7, 1, 3),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 7, 1, 3),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, 1),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 3, 1, 1),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, 1),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, 1, 1),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_16 = self.conv_1(x)

        x_32 = self.conv_2(F.max_pool1d(x_16, 4, 4))

        x_64 = self.conv_3(F.max_pool1d(x_32, 4, 4))

        x_128 = self.conv_4(F.max_pool1d(x_64, 4, 4))

        return x_16, x_32, x_64, x_128


class Task(nn.Module):
    def __init__(self):
        super(Task, self).__init__()
        self.block_1 = nn.Sequential(
            # TLAM(2100),
            CLAM(16),
            nn.Conv1d(16, 32, 1, 1),
            # nn.BatchNorm1d(32),
            # nn.ReLU(inplace=True)
            nn.MaxPool1d(4, 4)
        )
        self.block_2 = nn.Sequential(
            # TLAM(2100),
            CLAM(32),
            nn.Conv1d(32, 64, 1, 1),
            # nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 4)
        )

        self.block_3 = nn.Sequential(
            # TLAM(420),
            CLAM(64),
            nn.Conv1d(64, 128, 1, 1),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 4)
        )

        self.block_4 = nn.Sequential(
            CLAM(128),
            nn.Conv1d(128, 256, 1, 1),
            # nn.BatchNorm1d(256),
            # nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2)
        )

    def forward(self, x_16, x_32, x_64, x_128):
        x_32_a = self.block_1(x_16) + x_32
        x_64_a = self.block_2(x_32_a) + x_64
        x_128_a = self.block_3(x_64_a) + x_128
        x_256_a = self.block_4(x_128_a)
        return x_256_a


class MetaStress(nn.Module):
    def __init__(self, classes_num=2, hidden_size=128, num_layers=2):
        super(MetaStress, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fs = FS_Network()

        self.state_task = Task()
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, classes_num),
        )

        self.hr_task = Task()
        self.hr_estimator = nn.Linear(256, 1)

    def forward(self, x, net_type="both"):
        x_16, x_32, x_64, x_128 = self.fs(x)

        if net_type == "hr":
            hr_out = self.hr_task(x_16, x_32, x_64, x_128)
            hr = self.hr_estimator(F.adaptive_avg_pool1d(hr_out, 1).squeeze(-1))
        else:
            state_out = self.state_task(x_16, x_32, x_64, x_128)

            hr_out = self.hr_task(x_16, x_32, x_64, x_128)

            out = self.classifier(F.adaptive_avg_pool1d(state_out, 1).squeeze(-1))

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

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    print(get_parameter_number(net))
