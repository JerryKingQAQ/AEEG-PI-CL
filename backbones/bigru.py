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

class BiGRU(nn.Module):

    def __init__(self, hidden_dim, num_layers, embed_dim, dropout, num_classes):
        super(BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = hidden_dim * 2
        self.dropout_rate = dropout

        # gru
        self.bigru = nn.GRU(embed_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        #  dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        fre = x.shape[-1]
        x = F.avg_pool2d(x, kernel_size=(1, fre))
        x = x.view(x.size(0), x.size(1), -1)
        # gru
        gru_out, _ = self.bigru(x)
        gru_out = torch.transpose(gru_out, 1, 2)

        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2))
        gru_out = gru_out.squeeze(2)
        gru_out = self.dropout(gru_out)
        return gru_out

class CNN_BiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = SpaCNN(channels=32,channel_rate=8)
        self.bigru = BiGRU(hidden_dim=128, num_layers=2, embed_dim=32, num_classes=10, dropout=0)

    def forward(self,x):
        x = self.conv(x)
        out = self.bigru(x)
        return {
            'features': out
        }


if __name__ == '__main__':
    x = torch.randn((8, 32, 375, 6)).float()
    model = CNN_BiGRU()
    out = model(x)
    # print(out.shape)
    print(out['features'].shape)
