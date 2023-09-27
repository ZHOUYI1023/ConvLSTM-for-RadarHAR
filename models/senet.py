import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
   

class SENet(nn.Module):
    def __init__(self, num_classes=9, lstm_type='plain'):
        super(SENet, self).__init__()
        self.lstm_type = lstm_type
        self.conv1 = nn.Sequential(
            # input: batch, in_channel, length 
            # conv1d: in_channel, out_channel, kernel, stride, padding
            # size: (in_size-kernel+2*padding)/stride + 1
            # l_out: batch, out_channel, length
            nn.Conv1d(224, 32, 4, 2, 1),  # out: batch * 32 * 112
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),  # batch * 32 * 56
            nn.Conv1d(32, 64, 3, 1, 1),  # out: batch * 64 * 56
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 1, 1),  # batch * 64 * 56
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=3, batch_first=True, bidirectional=True)
        self.attention = Attention1D(in_channel=64)
        self.fc = nn.Linear(64, num_classes)
        self.fc1 = nn.Linear(384, num_classes)
        self.fcn_lstm = FCN_model(input_size=64)
        

    def forward(self, x):
        # input x: batch * 1 * 224 * 224
        y = x.squeeze(1)  # batch * 224 * 224
        y = self.conv1(y)  # batch * 64 * 112
        y = y.transpose(2, 1)  # batch * 112 * 64
        #lstm in: batch, length, feature_in
        #lstm out: batch, length, feature_out * 2(bi)
   
        if self.lstm_type == 'plain':
            out, hidden = self.lstm(y)  # out: batch * 112 * 64
            out = out[:, -1, :] # out: batch * 64
            out = self.fc(out)
        elif self.lstm_type == 'attention':
            out, hidden = self.lstm(y)  # out: batch * 112 * 64
            out, _ = self.attention(out) # out: batch * 64
            #print(out.shape)
            out = self.fc(out)

        elif self.lstm_type == 'fcn':
            out = self.fcn_lstm(y)
            out = self.fc1(out)
        
        return out
    