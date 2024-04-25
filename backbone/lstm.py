# -*- coding = utf-8 -*-
# @File : lstm.py
# @Software : PyCharm
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.out_dim = hidden_size
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.size(0), x.size(1), -1))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        # out = self.fc(out[:, -1, :])
        # return out
        return {
            'features': out
        }


if __name__ == '__main__':
    x = torch.randn((8, 32, 375, 6))
    model = LSTM(input_size=192, hidden_size=256, num_layers=3, num_classes=10)
    out = model(x)
    print(out['features'].shape)
