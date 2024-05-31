import torch
import torch.nn as nn
import torch.nn.functional as F


class CGRU(nn.Module):

    def __init__(self, hidden_dim, num_layers, embed_dim, conv_channel, kernel_size, num_classes, channel, dropout):
        super(CGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.channel = channel
        self.conv_channel = conv_channel
        self.out_dim = hidden_dim

        # CNN
        self.conv1 = nn.Conv2d(self.channel, self.conv_channel, self.kernel_size, padding=8)
        self.conv2 = nn.Conv2d(self.conv_channel, self.channel, self.kernel_size, padding=8)

        # GRU
        self.gru = nn.GRU(self.embed_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # CNN
        cnn_out = self.conv1(x)
        cnn_out = self.conv2(cnn_out)

        cnn_out = cnn_out.permute(0, 2, 1, 3)
        cnn_out = cnn_out.reshape((cnn_out.size(0), cnn_out.size(1), -1))

        # GRU

        gru_out, _ = self.gru(cnn_out)
        gru_out = torch.transpose(gru_out, 1, 2)

        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2))
        gru_out = gru_out.squeeze(2)
        gru_out = self.dropout(gru_out)
        return {
            'features': gru_out
        }


if __name__ == '__main__':
    x = torch.randn((8, 32, 375, 6)).float()
    model = CGRU(hidden_dim=192, num_layers=3, embed_dim=192, conv_channel=16, kernel_size=17, num_classes=10,
                 channel=32, dropout=0.1)
    out = model(x)

    print(out['features'].shape)
