# -*- coding = utf-8 -*-
# @File : CADCNN.py
# @Software : PyCharm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Channel_Attention(nn.Module):
    def __init__(self, channels, channel_rate=1):
        super().__init__()

        mid_channels = channels // channel_rate

        self.channel_attention = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # channel attention
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        return x


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class CADCNN(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck,dropout_rate):
        super(CADCNN, self).__init__()
        self.Channel_Attention = Channel_Attention(channels=32, channel_rate=4)
        self.out_dim = 104
        self.dropout = nn.Dropout(p=dropout_rate)
        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(32, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.Channel_Attention(x)
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.avg_pool2d(F.relu(self.bn1(out)), 1)
        out = torch.squeeze(out)
        out = F.avg_pool1d(out, out.size(2))
        out = torch.squeeze(out)
        out = self.dropout(out)
        return {
            'features': out
        }


if __name__ == '__main__':
    x = torch.randn((8, 32, 375, 6)).float()
    model = CADCNN(growthRate=16, depth=32, reduction=0.4,
                   bottleneck=True, nClasses=10,dropout_rate=0.1)
    out = model(x)
    # print(out.shape)
    print(out['features'].shape)
