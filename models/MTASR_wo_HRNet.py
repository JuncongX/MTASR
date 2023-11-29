import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.MTASR import Task, BVPFeatrueExtraction
import models.setting as setting


class MetaStress(nn.Module):
    def __init__(self, classes_num=2, hidden_size=128, num_layers=2):
        super(MetaStress, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bvp = BVPFeatrueExtraction()
        self.state_task = Task()
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, classes_num),
        )

    def forward(self, x):
        x_16, x_32, x_64_1, x_64_2, peak = self.bvp(x)
        state_out = self.state_task(x_16, x_32, x_64_1, x_64_2)
        out = self.classifier(F.adaptive_avg_pool1d(state_out, 1).squeeze(-1))
        return out, peak


if __name__ == '__main__':
    net = MetaStress()
    in_p = torch.randn((5, 1, 2100))
    out, peak = net(in_p)
    print(out.shape)
    print(peak.shape)
