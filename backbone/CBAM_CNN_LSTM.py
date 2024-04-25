import torch
import torch.nn as nn
import torch.nn.functional as F


class SpaCNN(nn.Module):
    def __init__(self, channels, channel_rate=1):
        super().__init__()
        mid_channels = channels // channel_rate

        self.spa_cnn = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # spatial attention
        x_spatial_att = self.spa_cnn(x).sigmoid()
        x = x * x_spatial_att
        return x


class ChaCNN(nn.Module):
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


class BiLSTM(nn.Module):

    def __init__(self, hidden_dim, num_layers, embed_dim, dropout):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = hidden_dim * 2
        self.dropout_rate = dropout

        # gru
        self.bilstm = nn.LSTM(embed_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                              batch_first=True)
        #  dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        fre = x.shape[-1]
        x = F.avg_pool2d(x, kernel_size=(1, fre))
        x = x.view(x.size(0), x.size(1), -1)
        # lstm
        lstm_out, _ = self.bilstm(x)
        lstm_out = torch.transpose(lstm_out, 1, 2)

        # pooling
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2))
        lstm_out = lstm_out.squeeze(2)
        lstm_out = self.dropout(lstm_out)
        return lstm_out


class CBAM_BiLSTM(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.spaconv = SpaCNN(channels=32, channel_rate=8)
        self.chaconv = ChaCNN(channels=32, channel_rate=8)
        self.bilstm = BiLSTM(hidden_dim=out_dim // 2, num_layers=2, embed_dim=32, dropout=0)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.spaconv(x)
        x = self.chaconv(x)
        out = self.bilstm(x)
        return {
            'features': out
        }


if __name__ == '__main__':
    x = torch.randn((8, 32, 375, 6)).float()
    model = CBAM_BiLSTM(out_dim=256)
    out = model(x)
    # print(out.shape)
    print(out['features'].shape)
