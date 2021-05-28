import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_util import ModelUtil


class BasicBlock(nn.Module):
    def __init__(
        self, in_planes, out_planes, drop_rate=0.0, norm_function=nn.BatchNorm2d
    ):
        super().__init__()
        self.bn1 = norm_function(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(
        self, in_planes, out_planes, drop_rate=0.0, norm_function=nn.BatchNorm2d
    ):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = norm_function(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = norm_function(inter_planes)
        self.conv2 = nn.Conv2d(
            inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop_rate > 0:
            out = F.dropout(
                out, p=self.drop_rate, inplace=False, training=self.training
            )
        out = self.conv2(self.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(
                out, p=self.drop_rate, inplace=False, training=self.training
            )
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(
        self, in_planes, out_planes, drop_rate=0.0, norm_function=nn.BatchNorm2d
    ):
        super(TransitionBlock, self).__init__()
        self.bn1 = norm_function(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop_rate > 0:
            out = F.dropout(
                out, p=self.drop_rate, inplace=False, training=self.training
            )
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(
        self,
        nb_layers,
        in_planes,
        growth_rate,
        block,
        drop_rate=0.0,
        norm_function=nn.BatchNorm2d,
    ):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, growth_rate, nb_layers, drop_rate, norm_function
        )

    def _make_layer(
        self, block, in_planes, growth_rate, nb_layers, drop_rate, norm_function
    ):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    in_planes + i * growth_rate,
                    growth_rate,
                    drop_rate,
                    norm_function=norm_function,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet3(nn.Module):
    def __init__(
        self,
        depth,
        num_classes,
        channels=3,
        growth_rate=12,
        reduction=0.5,
        bottleneck=True,
        drop_rate=0.0,
        norm_function=nn.BatchNorm2d,
    ):
        super().__init__()
        self.fuser_methods = list()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(
            channels, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = DenseBlock(
            n, in_planes, growth_rate, block, drop_rate, norm_function=norm_function
        )
        for name_list in ModelUtil(self.block1).get_sub_module_blocks(
            {(nn.Conv2d, nn.BatchNorm2d)}
        ):
            self.fuser_methods.append(["block1." + a for a in name_list])

        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(
            in_planes,
            int(math.floor(in_planes * reduction)),
            drop_rate=drop_rate,
            norm_function=norm_function,
        )
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.block2 = DenseBlock(
            n, in_planes, growth_rate, block, drop_rate, norm_function=norm_function
        )
        for name_list in ModelUtil(self.block2).get_sub_module_blocks(
            {(nn.Conv2d, nn.BatchNorm2d)}
        ):
            self.fuser_methods.append(["block2." + a for a in name_list])
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(
            in_planes,
            int(math.floor(in_planes * reduction)),
            drop_rate=drop_rate,
            norm_function=norm_function,
        )
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.block3 = DenseBlock(
            n, in_planes, growth_rate, block, drop_rate, norm_function=norm_function
        )
        for name_list in ModelUtil(self.block3).get_sub_module_blocks(
            {(nn.Conv2d, nn.BatchNorm2d)}
        ):
            self.fuser_methods.append(["block3." + a for a in name_list])
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = norm_function(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fuser_methods.append(["bn1", "relu"])
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2.0 / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)

    def get_fused_modules(self):
        return self.fuser_methods


def DenseNet40(num_classes, channels, **kwargs):
    return DenseNet3(40, num_classes=num_classes, channels=channels, **kwargs)
