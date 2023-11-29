# A social emotion classification approach using multi-model fusion
# Video-Based Depression Level Analysis by Encoding Deep Spatiotemporal Features

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
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(4608, 512)
        self.fc7 = nn.Linear(512, 512)
        # self.fc8 = nn.Linear(4096, 487)

        # self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

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

        h = h.view(batch_size, -1)
        h = self.fc6(h)
        h = self.relu(self.fc7(h))
        return h


class LSTM_anno(nn.Module):
    def __init__(self):
        super(LSTM_anno, self).__init__()
        # defining encoder LSTM layers
        self.rnn = nn.LSTM(512, 256, 1, batch_first=True)
        self.fc_final_score = nn.Linear(256, 2)

    def forward(self, x):
        self.rnn.flatten_parameters()
        state = None
        lstm_output, state = self.rnn(x, state)
        final_score = self.fc_final_score(lstm_output[:, -1, :])
        return final_score


class C3DLSTM(nn.Module):
    def __init__(self):
        super(C3DLSTM, self).__init__()
        self.c3d = C3D()
        self.lstm = LSTM_anno()

    def forward(self, x):
        batch_size, C, frames, H, W = x.shape
        clip_feats = torch.Tensor([]).to(x.device)
        for i in np.arange(0, frames - 1, 32):
            clip = x[:, :, i:i + 32, :, :]
            clip_feats_temp = self.c3d(clip)
            clip_feats_temp.unsqueeze_(0)
            clip_feats_temp.transpose_(0, 1)
            clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)
        return self.lstm(clip_feats)


if __name__ == '__main__':
    imgs = torch.rand((1, 3, 32, 64, 64))
    c3d_lstm = C3DLSTM()
    print(c3d_lstm(imgs).shape)

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""
