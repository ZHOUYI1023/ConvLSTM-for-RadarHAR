import torch
import torch.nn as nn


class FCN_model(nn.Module):
    def __init__(self, input_size):
        super(FCN_model,self).__init__()
        
        self.fcn = nn.Sequential(
            nn.Conv1d(64,8,8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8,64,8),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=3, batch_first=True, bidirectional=True)
   

    def forward(self,x):
        
        # input x should be in size [B,T,F] , where B = Batch size
        #                                         T = Time sampels
        #                                         F = features
        
        x1, (ht,ct) = self.lstm(x)
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)
        x2 = self.fcn(x2)
        x2 = x2.squeeze(-1)
        x_all = torch.cat((x1,x2),dim=1)
        
        return x_all

    
class Conv1DLSTM_FCN(nn.Module):
    def __init__(self, num_classes=9):
        super(Conv1DLSTM_FCN, self).__init__()
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
        self.fc = nn.Linear(128, num_classes)
        self.fcn_lstm = FCN_model(input_size=64)
        

    def forward(self, x):
        # input x: batch * 1 * 224 * 224
        y = x.squeeze(1)  # batch * 224 * 224
        y = self.conv1(y)  # batch * 64 * 112
        y = y.transpose(2, 1)  # batch * 112 * 64
        #lstm in: batch, length, feature_in
        #lstm out: batch, length, feature_out * 2(bi)
        out = self.fcn_lstm(y)
        out = self.fc(out)
        
        return out
    