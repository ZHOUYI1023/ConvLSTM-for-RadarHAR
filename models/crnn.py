import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes=200, nh=64):
        super(CRNN, self).__init__()
        
        #self.cvd_flag = cvd_flag
        self.encoder = nn.Sequential(
            # input: batch, in_channel, h, w
            # conv2d: in_channel, out_channel, kernel, stride, padding
            # out_size: (in_size-kernel+2*padding)/stride + 1
            # maxpool2d: kernel, stride
            # out_size: (in_size-kernel)/stride + 1
            # out: batch, out_channel, h, w
            nn.Conv2d(3, 128, 5, 1, 1),  # batch * 128 * 94 * 99
            nn.MaxPool2d((5, 2),stride=(5, 2),ceil_mode=False),  # batch * 128 * 18 * 49
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(),


            nn.Conv2d(128, 128, 5, 1, 1),  # batch * 128 * 16 * 47
            nn.MaxPool2d((4, 2),stride=(4, 2),ceil_mode=False),   # batch * 128 * 4 * 23
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(),

            nn.Conv2d(128, 128, 5, 1, 1),  # batch * 128 * 2 * 21
            nn.MaxPool2d((2, 2),stride=(2, 2),ceil_mode=False), # batch * 128 * 1 * 10
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (7, 3), (1, 1), (0, 1)),  # 128 * 64 * 7 * 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 48, (8, 3), (4, 1), (2, 1)),  # 128 * 48 * 28 * 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 32, (4, 3), (2, 1), (1, 1)),  # 128 * 32 * 56 * 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, 2),  # 128 * 16 * 112 * 112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),  # 128 * 16 * 224 * 224
            nn.ReLU(inplace=True),
        )

        
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        
        self.fc = nn.Sequential(nn.Linear(64, num_classes))


    def forward(self, x):
        # input: batch, 3, 96, 101
        #if self.cvd_flag:
            #x_input = torch.fft.fft(x, dim=-1)
        #else:
            #x_input = x
        #x_input = self.encoder(x_input)
        #x_recon = self.decoder(x_input)
        
        x = self.encoder(x) # batch, channel , 1, width 
        x = x.squeeze(2) # batch, channel , width(length)
        x = x.permute(0, 2, 1) # batch, length, feature
        out, _  = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out#, x_recon
    

class FCN_model(nn.Module):
    def __init__(self, input_size):
        super(FCN_model,self).__init__()
        
        self.fcn = nn.Sequential(
            nn.Conv1d(input_size,128,2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256,128,2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
   

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

    
class CRNN_1(nn.Module):
    def __init__(self, num_classes=200, nh=64, lstm_type='plain'):
        super(CRNN_1, self).__init__()
        self.lstm_type = lstm_type
        #self.cvd_flag = cvd_flag
        self.encoder = nn.Sequential(
            # input: batch, in_channel, h, w
            # conv2d: in_channel, out_channel, kernel, stride, padding
            # out_size: (in_size-kernel+2*padding)/stride + 1
            # maxpool2d: kernel, stride
            # out_size: (in_size-kernel)/stride + 1
            # out: batch, out_channel, h, w
            nn.Conv2d(3, 128, 5, 1, 1),  # batch * 128 * 94 * 99
            nn.MaxPool2d((5, 2),stride=(5, 2),ceil_mode=False),  # batch * 128 * 18 * 49
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(),


            nn.Conv2d(128, 128, 5, 1, 1),  # batch * 128 * 16 * 47
            nn.MaxPool2d((4, 2),stride=(4, 2),ceil_mode=False),   # batch * 128 * 4 * 23
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(),

            nn.Conv2d(128, 128, 5, 1, 1),  # batch * 128 * 2 * 21
            nn.MaxPool2d((2, 2),stride=(2, 2),ceil_mode=False), # batch * 128 * 1 * 10
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (7, 3), (1, 1), (0, 1)),  # 128 * 64 * 7 * 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 48, (8, 3), (4, 1), (2, 1)),  # 128 * 48 * 28 * 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 32, (4, 3), (2, 1), (1, 1)),  # 128 * 32 * 56 * 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, 2),  # 128 * 16 * 112 * 112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),  # 128 * 16 * 224 * 224
            nn.ReLU(inplace=True),
        )

        
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        
        self.fc = nn.Sequential(nn.Linear(64, num_classes))
        
        self.attention = Attention1D(in_channel=64)
        self.fc1 = nn.Linear(384, num_classes)
        self.fcn_lstm = FCN_model(input_size=128)
        
            


    def forward(self, x):
        # input: batch, 3, 96, 101
        #if self.cvd_flag:
            #x_input = torch.fft.fft(x, dim=-1)
        #else:
            #x_input = x
        #x_input = self.encoder(x_input)
        #x_recon = self.decoder(x_input)
        
        x = self.encoder(x) # batch, channel , 1, width 
        x = x.squeeze(2) # batch, channel , width(length)
        x = x.permute(0, 2, 1) # batch, 10, 128
        if self.lstm_type == 'plain':
            out, _ = self.lstm(x)  # out: batch * 10 * 64
            out = out[:, -1, :] # out: batch * 64
            out = self.fc(out)
        elif self.lstm_type == 'attention':
            out, _ = self.lstm(x)  # out: batch * 10 * 64
            out, _ = self.attention(out) # out: batch * 64
            #print(out.shape)
            out = self.fc(out)

        elif self.lstm_type == 'fcn':
            out = self.fcn_lstm(x)
            out = self.fc1(out)
            
        return out#, x_recon