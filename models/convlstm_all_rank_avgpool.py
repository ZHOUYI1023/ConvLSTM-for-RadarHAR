import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair


def flip_half(output):
    B, C, T = output.size()
    half1 = output[:, :C//2, :]
    half2 = output[:, C//2:, :]
    half2_flipped = torch.flip(half2, dims=[1])
    output = torch.cat((half1, half2_flipped), dim=1)
    return output


class SymmetricLightweightConv(nn.Module):
    def __init__(self, d_size, groups=2, kernel_size=3, padding=1, n_heads=1,
                 weight_softmax=True, bias=False, dropout=0.0):
        super().__init__()
        self.input_size = d_size
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.padding = padding
        self.groups = groups
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(np.int(n_heads*self.input_size/2), 1, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(self.input_size)) if bias else None
        self.dropout = dropout
        self.weight_dropout = 0.1
        self.training = True

    def forward(self, input):
        B, C, T = input.size()
        input = flip_half(input)
        reshaped_tensor = input.view(B * 2, C // 2, T)
        #print(reshaped_tensor.shape)
        H = self.n_heads
        # weight: n_heads, 1, kernel_size 
        #normalized_weights = (self.weight - torch.mean(self.weight)) / torch.std(self.weight)
        weight = F.softmax(self.weight, dim=-1) if self.weight_softmax else self.weight
        weight = F.dropout(weight, self.weight_dropout, training=self.training)
        #print(weight.shape)
        output = F.conv1d(reshaped_tensor, weight, padding=self.padding, groups=np.int(self.groups/2))
        #print(output.shape)
        output = output.view(B, C, T)
        output = flip_half(output)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output


    
class Conv1DEncoder(nn.Module):
    def __init__(self, d_size):
        super(Conv1DEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            # input: batch, in_channel, length 
            # conv1d: in_channel, out_channel, kernel, stride, padding
            # size: (in_size-kernel+2*padding)/stride + 1
            # l_out: batch, out_channel, length

            nn.Conv1d(d_size, 32, 5, 1, 2),  # out: batch * 32 * 112
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # batch * 32 * 56

        )
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 3, 1, 1),  # out: batch * 64 * 56
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 3, 1, 1),  # out: batch * 64 * 56
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        

    def forward(self, x):
        # input x:  batch * C * T
        y = self.conv1(x)  # batch * 64 * 112
        #print(y.shape)
        y = self.conv2(y)
        #y = self.conv3(y)
        
        return y
    
    
class Attention1D(nn.Module):
    def __init__(self, in_channel):
        super(Attention1D, self).__init__()
        self.tanh = nn.Tanh()
        self.weight = nn.Linear(in_channel,1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        M = self.tanh(H)  # (batch, seq_len, rnn_size)
        alpha = self.weight(M).squeeze(2)  # (batch, seq_len)
        alpha = self.softmax(alpha)  # (batch, seq_len)

        r = H * alpha.unsqueeze(2) # (batch, seq_len, rnn_size)
        r = r.sum(dim=1)  # (batch, rnn_size)

        return r, alpha 

    
class Attentional_LSTM_Pool(nn.Module):
    def __init__(self, d_size):
        super(Attentional_LSTM_Pool,self).__init__()
        
        self.attention = Attention1D(in_channel=64)
        self.lstm = nn.LSTM(input_size=d_size, hidden_size=32, num_layers=3, batch_first=True, bidirectional=True)

    def forward(self,x):
        
        # x:  B * T * C
        x1, (ht,ct) = self.lstm(x) # x1: B, T, bi*hidden_size
        x1, _ = self.attention(x1) # out: batch, bi*hidden_size: 64
        x2 = torch.max(x, 1, keepdim=False)[0] #  B * C
        out = torch.cat((x1,x2),dim=1)        
        return out, x1


class Conv1DLSTM_All(nn.Module):
    def __init__(self, d_size, t_size, num_classes=9, weight_softmax=False):
        super(Conv1DLSTM_All, self).__init__()
        self.d_size = d_size
        self.t_size = t_size
        self.conv0 = SymmetricLightweightConv(d_size=self.d_size,groups=self.d_size, kernel_size=3, padding=1, n_heads=1, weight_softmax=True)
        self.pool = nn.AdaptiveAvgPool1d(self.t_size)
        self.conv1 = Conv1DEncoder(d_size = self.d_size)
        self.attention_lstm_pool = Attentional_LSTM_Pool(d_size = 64)
        self.fc = nn.Linear(64+64, num_classes)


    def forward(self, x):
        # input x: batch * 1 * 224 * 224
        y = x.squeeze(1)  # batch * C * T
        y = self.conv0(y)
        y = self.pool(y) # batch * C * T
        y = self.conv1(y)  # batch * C * T
        y = y.transpose(2, 1)  # batch * T(64) * C(128)
        out, f_st = self.attention_lstm_pool(y)  # out: batch * 112 * 64
        out = self.fc(out)
        
        return out, f_st, y