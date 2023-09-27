import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair


def _spectral_crop(input, oheight, owidth):
    cutoff_freq_h = math.ceil(oheight / 2)
    cutoff_freq_w = math.ceil(owidth / 2)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            bottom_left = input[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

    top_combined = torch.cat((top_left, top_right), dim=-1)
    bottom_combined = torch.cat((bottom_left, bottom_right), dim=-1)
    all_together = torch.cat((top_combined, bottom_combined), dim=-2)

    return all_together

def _spectral_pad(input, output, oheight, owidth):
    cutoff_freq_h = math.ceil(oheight / 2)
    cutoff_freq_w = math.ceil(owidth / 2)
    pad = torch.zeros_like(input)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            pad[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):] = output[:, :, -(cutoff_freq_h-1):, -(cutoff_freq_w-1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h-1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:] = output[:, :, -(cutoff_freq_h-1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w-1):]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):] = output[:, :, -cutoff_freq_h:, -(cutoff_freq_w-1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -cutoff_freq_w:] = output[:, :, -cutoff_freq_h:, -cutoff_freq_w:]	

    return pad

def DiscreteHartleyTransform(input):
    fft = torch.rfft(input, 2, normalized=True, onesided=False)
    # for new version of pytorch
    #fft = torch.fft.fft2(input, dim=(-2, -1), norm='ortho')
    #fft = torch.stack((fft.real, fft.imag), -1)
    dht = fft[:, :, :, :, -2] - fft[:, :, :, :, -1]
    return dht

class SpectralPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, oheight, owidth):
        ctx.oh = oheight
        ctx.ow = owidth
        ctx.save_for_backward(input)

        # Hartley transform by RFFT
        dht = DiscreteHartleyTransform(input)

        # frequency cropping
        all_together = _spectral_crop(dht, oheight, owidth)
        # inverse Hartley transform
        dht = DiscreteHartleyTransform(all_together)
        return dht

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables

        # Hartley transform by RFFT
        dht = DiscreteHartleyTransform(grad_output)
        # frequency padding
        grad_input = _spectral_pad(input, dht, ctx.oh, ctx.ow)
        # inverse Hartley transform
        grad_input = DiscreteHartleyTransform(grad_input)
        return grad_input, None, None

class SpectralPool2d(nn.Module):
    def __init__(self, t_size):
        super(SpectralPool2d, self).__init__()
        self.t_size = t_size
    def forward(self, input):
        H, W = input.size(-2), input.size(-1)
        #h, w = math.ceil(H*self.scale_factor[0]), math.ceil(W*self.scale_factor[1])
        return SpectralPoolingFunction.apply(input, H, self.t_size)



class SpectralPooling_layer(nn.Module):
    def __init__(self, t_size):
        super(SpectralPooling_layer, self).__init__()
        self.t_size = t_size
        self.SpecPool2d = SpectralPool2d(t_size=t_size)

    def forward(self, x):
        # input: batch, in_channel, length 
        x = x.unsqueeze(1)   # input: batch, 1, in_channel, length 
        out = self.SpecPool2d(x)
        return out.squeeze()

    
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

        self.fcn = nn.Sequential(
            nn.Conv1d(d_size,8,8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8,d_size,8),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self,x):
        
        # x:  B * T * C
        x1, (ht,ct) = self.lstm(x) # x1: B, T, bi*hidden_size
        x1, _ = self.attention(x1) # out: batch, bi*hidden_size: 64
        return x1, x1


class Conv1DLSTM_All(nn.Module):
    def __init__(self, d_size, t_size, num_classes=9, weight_softmax=False):
        super(Conv1DLSTM_All, self).__init__()
        self.d_size = d_size
        self.t_size = t_size
        self.conv0 = SymmetricLightweightConv(d_size=self.d_size,groups=self.d_size, kernel_size=3, padding=1, n_heads=1, weight_softmax=True)
        self.pool = SpectralPooling_layer(self.t_size)
        self.conv1 = Conv1DEncoder(d_size = self.d_size)
        self.attention_lstm_pool = Attentional_LSTM_Pool(d_size = 64)
        self.fc = nn.Linear(64, num_classes)


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
    
    
    