# Howard A, Sandler M, Chu G, et al. Searching for mobilenetv3[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2019: 1314-1324.
# https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU6(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, f, h, w = x.size()
        y = F.avg_pool3d(x, x.data.size()[-3:]).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return (x * y)


def conv_3x3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        self.use_res_connect = self.stride == (1,1,1) and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                h_swish() if use_hs else nn.ReLU6(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                h_swish() if use_hs else nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, sample_size=224, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        assert sample_size % 16 == 0.
        input_channel = _make_divisible(16 * width_mult, 8)
        self.features = [conv_3x3x3_bn(3, input_channel, (1, 2, 2))]

        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, se, hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            hidden_dim = _make_divisible(input_channel * t, 8)
            self.features.append(block(input_channel, hidden_dim, output_channel, k, s, se, hs))
            input_channel = output_channel

        self.features.append(conv_1x1x1_bn(input_channel, hidden_dim))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[
            mode]

        # building classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,   1,  16, 0, 0, (1, 1, 1)],
        [3,   4,  24, 0, 0, (2, 2, 2)],
        [3,   3,  24, 0, 0, (1, 1, 1)],
        [5,   3,  40, 1, 0, (2, 2, 2)],
        [5,   3,  40, 1, 0, (1, 1, 1)],
        [5,   3,  40, 1, 0, (1, 1, 1)],
        [3,   6,  80, 0, 1, (2, 2, 2)],
        [3, 2.5,  80, 0, 1, (1, 1, 1)],
        [3, 2.3,  80, 0, 1, (1, 1, 1)],
        [3, 2.3,  80, 0, 1, (1, 1, 1)],
        [3,   6, 112, 1, 1, (1, 1, 1)],
        [3,   6, 112, 1, 1, (1, 1, 1)],
        [5,   6, 160, 1, 1, (2, 2, 2)],
        [5,   6, 160, 1, 1, (1, 1, 1)],
        [5,   6, 160, 1, 1, (1, 1, 1)]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,    1,  16, 1, 0, (2, 2, 2)],
        [3,  4.5,  24, 0, 0, (2, 2, 2)],
        [3, 3.67,  24, 0, 0, (1, 1, 1)],
        [5,    4,  40, 1, 1, (2, 2, 2)],
        [5,    6,  40, 1, 1, (1, 1, 1)],
        [5,    6,  40, 1, 1, (1, 1, 1)],
        [5,    3,  48, 1, 1, (1, 1, 1)],
        [5,    3,  48, 1, 1, (1, 1, 1)],
        [5,    6,  96, 1, 1, (2, 2, 2)],
        [5,    6,  96, 1, 1, (1, 1, 1)],
        [5,    6,  96, 1, 1, (1, 1, 1)],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


def get_model(**kwargs):
    """
    Returns the model.
    """
    return mobilenetv3_large(**kwargs)


if __name__ == "__main__":
    model = get_model(num_classes=2, sample_size=64, width_mult=1.)
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)
    # print(model)

    # input_var = Variable(torch.randn(2, 3, 2098, 64, 64))
    # output = model(input_var)
    # print(output.shape)

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    print(get_parameter_number(model))