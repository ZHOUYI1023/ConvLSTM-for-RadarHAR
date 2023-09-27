import torch
import torch.nn as nn


class FCN_model(nn.Module):
    def __init__(self, input_size):
        super(FCN_model,self).__init__()
        
        self.fcn = nn.Sequential(
            nn.Conv1d(input_size,128,8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,5),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256,128,3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
   

    def forward(self,x):
        
        # input x should be in size [B,T,F] , where B = Batch size
        #                                         T = Time sampels
        #                                         F = features
        
        x1, (ht,ct) = self.lstm(x)
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)
        x2 = self.fcn(x2)
        x2 = torch.mean(x2,2)
        
        x_all = torch.cat((x1,x2),dim=1)
        
        return x_all


class Attention1D(nn.Module):
    def __init__(self, in_channel:int):
        super(Attention1D, self).__init__()
        self.tanh = nn.Tanh()
        self.weight = nn.Linear(in_channel,1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        M = self.tanh(H)  # (batch, seq_len, rnn_size)  seq_len可以理解为时间维 rnn_size为lstm输入
        alpha = self.weight(M).squeeze(2)  # (batch, seq_len)
        alpha = self.softmax(alpha)  # (batch, seq_len)

        r = H * alpha.unsqueeze(2) # (batch, seq_len, rnn_size)
        r = r.sum(dim=1)  # (batch, rnn_size)

        return r, alpha
    
class Conv1DLSTM(nn.Module):
    def __init__(self, num_classes=9, lstm_type='plain'):
        super(Conv1DLSTM, self).__init__()
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
    