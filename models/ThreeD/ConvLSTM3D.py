# End-to-End Continuous Emotion Recognition from Video Using 3D Convlstm Networks

import torch
import torch.nn as nn
import numpy as np
from models.ThreeD.ConvLSTM import ConvLSTM


class ConvLSTM3D(nn.Module):
    def __init__(self):
        super(ConvLSTM3D, self).__init__()
        # nn.Conv3d(input_channel, k, kernel_size=(d, f, f), padding=(), stride=(st, sp, sp))
        # nn.MaxPool3d(kernel_size=(d, f, f), stride=(st, sp, sp))
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # self.fc8 = nn.Linear(4096, 487)

        # self.dropout = nn.Dropout(p=0.5)
        # self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.conv_lstm = ConvLSTM(256, 64, (3, 3), 1, batch_first=True)  # B, T, C, H, W

        self.fc = nn.Sequential(
            nn.Linear(4096, 256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        batch_size, _, _, _, _ = x.shape
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3(h))
        h = self.pool3(h)

        h = h.transpose(1, 2)

        _, last_states = self.conv_lstm(h)

        out = self.fc(last_states[-1][-1].view(batch_size, -1))

        return out


if __name__ == '__main__':
    import os

    device_list = [0]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgs = torch.rand((1, 3, 2096, 64, 64)).to(device)
    convlstm3d = ConvLSTM3D().to(device)
    print(convlstm3d(imgs).shape)
