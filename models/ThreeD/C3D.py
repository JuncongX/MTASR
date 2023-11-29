# Learning Spatiotemporal Features With 3D Convolutional Networks

import torch
import torch.nn as nn
import numpy as np


class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        # nn.Conv3d(input_channel, k, kernel_size=(d, f, f), padding=(), stride=(st, sp, sp))
        # nn.MaxPool3d(kernel_size=(d, f, f), stride=(st, sp, sp))
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.AdaptiveMaxPool3d(1)

        self.fc6 = nn.Linear(512, 4096)
        self.fc7 = nn.Linear(4096, 2)
        # self.fc8 = nn.Linear(4096, 487)

        # self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, _, _, _ = x.shape
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)
        # print('Shape of pool5: ', h.shape)

        h = h.view(batch_size, -1)
        h = self.fc6(h)

        return self.fc7(h)

if __name__ == '__main__':
    imgs = torch.rand((1, 3, 2096, 64, 64)).cuda()
    c3d = C3D().cuda()
    print(c3d(imgs).shape)

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""
