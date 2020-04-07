#!/usr/bin/env python3
import math

import collections

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(
            collections.OrderedDict(
                [
                    ("c1", nn.Conv2d(1, 6, kernel_size=(5, 5))),
                    ("relu1", nn.ReLU()),
                    ("s2", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("c3", nn.Conv2d(6, 16, kernel_size=(5, 5))),
                    ("relu3", nn.ReLU()),
                    ("s4", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("c5", nn.Conv2d(16, 120, kernel_size=(5, 5))),
                    ("relu5", nn.ReLU()),
                ]
            )
        )

        self.fc = nn.Sequential(
            collections.OrderedDict(
                [
                    ("f6", nn.Linear(120, 84)),
                    ("relu6", nn.ReLU()),
                    ("f7", nn.Linear(84, 10)),
                    ("sig7", nn.LogSoftmax(dim=-1)),
                ]
            )
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(x.size(0), -1)
        output = self.fc(output)
        return output


class MobileNetV2CIFAR10(torchvision.models.MobileNetV2):
    def __init__(self):
        super(MobileNetV2CIFAR10, self).__init__(
            num_classes=10,
            inverted_residual_setting=[
                [1, 16, 1, 1],
                # [6, 24, 2, 2],
                [6, 24, 2, 1],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ],
        )
        self.features[0][0].stride = (1, 1)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            4 * growth_rate,
            kernel_size=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(
            self,
            block,
            nblocks,
            growth_rate=12,
            reduction=0.5,
            num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(
            3,
            num_planes,
            kernel_size=3,
            padding=1,
            bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def densenet_cifar():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12)


class SVHN(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(SVHN, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(n_channel, num_classes))
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=padding
            )
            if batch_norm:
                layers += [
                    conv2d,
                    nn.BatchNorm2d(out_channels, affine=False),
                    nn.ReLU(),
                ]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)


def stl10(n_channel):
    cfg = [
        n_channel,
        "M",
        2 * n_channel,
        "M",
        4 * n_channel,
        "M",
        4 * n_channel,
        "M",
        (8 * n_channel, 0),
        (8 * n_channel, 0),
        "M",
    ]
    layers = make_layers(cfg, batch_norm=True)
    model = SVHN(layers, n_channel=8 * n_channel, num_classes=10)
    return model
