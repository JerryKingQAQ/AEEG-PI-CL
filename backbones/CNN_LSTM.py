import torch
import torch.nn as nn
import torch.nn.functional as F


class CLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, embed_dim, conv_channel, kernel_size, num_classes, channel, dropout):
        super(CLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.channel = channel
        self.conv_channel = conv_channel
        self.out_dim = embed_dim

        # CNN
        self.conv1 = nn.Conv1d(self.channel, self.conv_channel, self.kernel_size, padding=15)
        self.conv2 = nn.Conv1d(self.conv_channel, self.channel, self.kernel_size, padding=15)

        # LSTM
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, num_layers=self.num_layers, bias=True)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.size(0), x.size(1), -1))

        # CNN
        cnn_out = x
        cnn_out = cnn_out.view(cnn_out.size(0), cnn_out.size(1), -1)
        cnn_out = self.conv1(cnn_out)
        cnn_out = self.conv2(cnn_out)
        # LSTM
        lstm_out, _ = self.lstm(cnn_out)

        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2))
        lstm_out = lstm_out.squeeze(2)
        lstm_out = self.dropout(lstm_out)

        return {
            'features': lstm_out
        }

if __name__ == '__main__':
    x = torch.randn((8, 32, 375, 6)).float()
    model = CLSTM(hidden_dim=192, num_layers=3, embed_dim=192, conv_channel=64, kernel_size=31, num_classes=10,
                 channel=375, dropout=0.1)
    out = model(x)
    # print(out.shape)
    print(out['features'].shape)
