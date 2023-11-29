# Multitask Learning Based on Lightweight 1DCNN for Fault Diagnosis of Wheelset Bearings
import torch
import torch.nn as nn
import torch.nn.functional as F


class MT_1DCNN(nn.Module):
    def __init__(self):
        super(MT_1DCNN, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 16, 12, 2, 5),
            nn.ReLU(),
            nn.Conv1d(16, 16, 12, 2, 5),
            nn.ReLU(),
            nn.Conv1d(16, 24, 9, 2, 3),
            nn.ReLU(),
            nn.Conv1d(24, 24, 9, 2, 3),
            nn.ReLU(),
            nn.Conv1d(24, 32, 6, 2, 2),
            nn.ReLU(),
        )

        self.hr_b = nn.Sequential(
            nn.Conv1d(32, 32, 6, 2, 2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, 2, 1),
            nn.ReLU(),
        )

        self.stress_b = nn.Sequential(
            nn.Conv1d(32, 32, 6, 2, 2),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, 2, 1),
            nn.ReLU()
        )

        self.stress = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        self.hr = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x, net_type="both"):
        x = self.conv_1(x)
        hr_f = self.hr_b(x)
        stress_f = self.stress_b(x)

        if net_type == "hr":
            return self.hr(hr_f)
        else:
            return self.stress(stress_f), self.hr(hr_f)


if __name__ == '__main__':
    net = MT_1DCNN()
    in_p = torch.rand((4, 1, 2100))
    out, hr = net(in_p)
    print(out.shape, hr.shape)

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    print(get_parameter_number(net))