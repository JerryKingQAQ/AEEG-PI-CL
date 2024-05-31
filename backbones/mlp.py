# -*- coding = utf-8 -*-
# @File : mlp.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.out_dim = output_size
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.size(0), x.size(1), -1))

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return {
            'features': x
        }


if __name__ == '__main__':
    x = torch.randn((8, 32, 375, 6)).float()
    model = MLP(input_size=192, hidden_size1=64, hidden_size2=248, output_size=256)
    out = model(x)
    # print(out.shape)
    print(out['features'].shape)
