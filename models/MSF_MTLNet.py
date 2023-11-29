# Reference: PPG-based blood pressure estimation can benefit from scalable multi-scale fusion neural networks and multi-task learning
import torch
import torch.nn as nn
import torch.nn.functional as F


class SKUnit(nn.Module):
    def __init__(self, in_channels, r=4, L=8, stride=1):
        super(SKUnit, self).__init__()
        d = max(int(in_channels / r), L)

        self.conv_k3 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.Hardswish(inplace=True)
        )

        self.conv_k5 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 3, padding=2, dilation=2, stride=stride, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.Hardswish(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, d, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(d),
            nn.Hardswish(inplace=True)
        )

        self.A = nn.Conv1d(d, in_channels, kernel_size=1, stride=1, bias=False)

        self.B = nn.Conv1d(d, in_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        U_3k = self.conv_k3(x)
        U_5k = self.conv_k5(x)
        U = U_3k + U_5k
        s = self.avg_pool(U)
        z = self.fc(s)

        eAz = torch.exp(self.A(z))
        eBz = torch.exp(self.B(z))

        a = eAz / (eAz + eBz)
        b = eBz / (eAz + eBz)

        V = a * U_3k + b * U_5k
        return V


class MSF_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MSF_Block, self).__init__()
        self.PW = nn.Sequential(
            nn.Conv1d(in_channels, 6 * in_channels, 1, 1),
            nn.BatchNorm1d(6 * in_channels),
            nn.Hardswish()
        )
        self.DW = nn.Sequential(
            SKUnit(6 * in_channels, stride=stride),
            nn.Conv1d(6 * in_channels, out_channels, 1, 1),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        pw = self.PW(x)
        dw = self.DW(pw)
        return self.shortcut(x) + dw


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BAM(nn.Module):
    def __init__(self, in_channels):
        super(BAM, self).__init__()
        self.ca = nn.Sequential(
            SELayer(in_channels),
            nn.BatchNorm1d(in_channels)
        )

        self.ta = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, 1, 1),
            nn.Conv1d(in_channels // 4, in_channels // 4, 3, dilation=4, padding=4),
            nn.Conv1d(in_channels // 4, in_channels // 4, 3, dilation=4, padding=4),
            nn.Conv1d(in_channels // 4, 1, 1, 1),
            nn.BatchNorm1d(1)
        )

    def forward(self, x):
        ca = self.ca(x)
        ta = self.ta(x)
        return ca * ta * x + x


class Subnetwork(nn.Module):
    def __init__(self):
        super(Subnetwork, self).__init__()
        self.conv_1 = nn.Sequential(
            BAM(128),
            nn.Conv1d(128, 768, 1, 1),
            nn.Hardswish()
        )

        self.conv_2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(768, 128, 1, 1),
            nn.Hardswish()
        )

    def forward(self, x):
        x_1 = self.conv_1(x)

        return self.conv_2(x_1)


class MSF_MTLNet(nn.Module):
    def __init__(self):
        super(MSF_MTLNet, self).__init__()
        self.Backbone = nn.Sequential(
            nn.Conv1d(3, 16, 1, 2),
            nn.Hardswish(),
            MSF_Block(16, 32, 2),
            MSF_Block(32, 32),
            MSF_Block(32, 32),
            MSF_Block(32, 32, 2),
            MSF_Block(32, 64),
            MSF_Block(64, 64, 2),
            MSF_Block(64, 128, 2),
        )

        self.subnetwork_1 = Subnetwork()

        self.hr = nn.Sequential(
            # nn.Conv1d(128, 1, 1, 1)
            nn.Linear(128, 1)
        )

        self.subnetwork_2 = Subnetwork()

        self.stress = nn.Sequential(
            # nn.Conv1d(128, 2, 1, 1)
            nn.Linear(128, 512),
            nn.Hardswish(),
            nn.Linear(512, 2)
        )

    def forward(self, x, net_type="both"):
        f = self.Backbone(x)
        f_1 = self.subnetwork_1(f)
        f_2 = self.subnetwork_2(f)
        if net_type == "hr":
            return self.hr(f_1.squeeze(-1))
        else:
            return self.stress(f_2.squeeze(-1)), self.hr(f_1.squeeze(-1))


if __name__ == '__main__':
    net = MSF_MTLNet()
    in_p = torch.rand((1, 3, 2100))
    out, hr = net(in_p)
    print(out.shape, hr.shape)

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    print(get_parameter_number(net))
