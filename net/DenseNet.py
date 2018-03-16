import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict


class Bottleneck(nn.Module):
    def __init__(self, in_plane, growth_rate, bn_size, drop_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_plane)
        self.conv1 = nn.Conv2d(in_plane, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        # connect out and x to [out,x]
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_plane)
        self.conv1 = nn.Conv2d(in_plane, out_plane, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseLayer(nn.Module):
    def __init__(self, in_plane, block_size, bn_size, growth_rate, drop_rate):
        super(DenseLayer, self).__init__()
        self.layers = []
        for i in range(block_size):
            self.layers.append(Bottleneck(in_plane + i * growth_rate, growth_rate, bn_size, drop_rate))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

        Args:
            growth_rate (int) - how many filters to add each layer (`k` in paper)
            block_config (list of 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
              (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
            num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()

        # First Convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2)),
            ('bn0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('maxpool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # DenseLayer
        num_features = num_init_features
        for i in range(len(block_config)):
            denseblock = DenseLayer(num_init_features/2,
                                    block_size=block_config[i],
                                    bn_size=bn_size,
                                    growth_rate=growth_rate,
                                    drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), denseblock)
            num_features += growth_rate

            if i != len(block_config) - 1:
                transition = Transition(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), transition)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
