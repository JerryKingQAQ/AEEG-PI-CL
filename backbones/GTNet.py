import torch
import torch.nn as nn
import torch.nn.functional as F


class GTNet(nn.Module):
    def __init__(self, lstm_hidden_dim, lstm_num_layers, embed_dim, num_classes, dropout):
        super(GTNet, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.num_layers = lstm_num_layers
        self.dropout_rate = dropout

        self.bilstm = nn.LSTM(embed_dim, self.hidden_dim // 2, num_layers=self.num_layers,
                              bidirectional=True, batch_first=True)

        self.out_dim = lstm_hidden_dim
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.size(0), x.size(1), -1))
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = self.dropout(bilstm_out)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2))
        bilstm_out = bilstm_out.squeeze(2)
        out = self.gelu(bilstm_out)
        return {
            'features': out
        }


if __name__ == '__main__':
    x = torch.randn((8, 32, 375, 6)).float()
    model = GTNet(lstm_hidden_dim=256, lstm_num_layers=2, embed_dim=192, num_classes=10, dropout=0.1)
    out = model(x)
    # print(out.shape)
    print(out['features'].shape)
