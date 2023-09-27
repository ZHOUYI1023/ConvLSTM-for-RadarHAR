import torch
import torch.nn as nn
import torch.nn.functional as F

        
        
class LightweightConv(nn.Module):
    '''Lightweight convolution from fairseq.
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_softmax: normalize the weight with softmax before the convolution
        dropout: dropout probability
    Forward:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)
    Attributes:
        weight: learnable weights of shape `(num_heads, 1, kernel_size)`
        bias:   learnable bias of shape `(input_size)`
    '''
    def __init__(self, input_size, kernel_size=1, padding=0, n_heads=1,
                 weight_softmax=True, bias=False, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(n_heads, 1, kernel_size))
        self.weight_linear = nn.Linear(input_size, n_heads * kernel_size, bias)
        self.bias = nn.Parameter(torch.Tensor(input_size)) if bias else None
        self.dropout = dropout
        self.weight_dropout = 0.1
        self.training = True

    def forward(self, input):
        '''Takes input (B x C x T) to output (B x C x T)'''
        
        # Prepare weight (take softmax)
        B, C, T = input.size()
        H, K = self.n_heads, self.kernel_size
        # weight: n_heads, 1, kernel_size 
        weight = self.weight_linear(input.permute(0,2,1)).view(B, T, H, K)
        weight = F.softmax(weight, dim=-1) 
        #weight = F.softmax(self.weight, dim=-1) if self.weight_softmax else self.weight
        weight = F.dropout(weight, self.weight_dropout, training=self.training)
        weight = weight.permute(0, 2, 3, 1)
        weight = weight.contiguous().view(-1, H, T)
        print(weight.shape)
        # Merge every C/H entries into the batch dimension (C = self.input_size)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.contiguous().view(-1, H, T)
        # input_tensor, kernel, stride=1, padding=1
        output = F.conv1d(input, weight, padding=self.padding, groups=H)
        output = output.view(B, C, T)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output
    
    
class ConvBlock(nn.Module):
    """Lightweight or dynamic convolutional layer"""
    def __init__(self, cfg, kernel_size):
        self.norm_before_conv = cgf.norm_before_conv
        
        # Initial fully connected layer or GLU
        self.linear_1 = Linear(cfg.dim, cfg.dim * (2 if cfg.glu_in_conv else 1))
        self.glu = nn.GLU() if cfg.glu_in_conv else None

        # Lightweight or dynamic convolution
        assert cfg.conv_type in ['lightweight', 'dynamic']
        Conv = LightweightConv if cfg.conv_type == 'lightweight' else DynamicConv
        self.conv = Conv(cfg.dim, kernel_size=kernel_size, padding_l=kernel_size-1, # amount of padding
                         weight_softmax=cfg.weight_softmax, n_heads=n_heads, dropout=cfg.p_drop_conv)

        # I do not think this second linear layer is necessary, but we will do it anyway
        self.linear_2 = nn.Linear(cfg.dim, cfg.dim)

        # Dropout and layer normalization
        self.dropout = nn.Dropout(cfg.p_drop_hidden)
        self.conv_layer_norm = LayerNorm(cfg.dim)
        
        # NOTE: This is where the encoder attention would go if there were any

        # Final linear layer: See Figure 2 in the LWDC paper
        self.fc1 = Linear(cfg.dim, cgf.dim_ff)
        self.fc2 = Linear(cgf.dim_ff, cfg.dim)
        self.final_layer_norm = LayerNorm(cfg.dim)

    def __forward__(self, cfg):
        '''See Figure 2(b) in the paper'''
        
        # Linear and GLU
        res = x
        if self.norm_before_conv: 
            x = self.conv_layer_norm(x)
        x = self.dropout(self.linear_1(x))
        x = x if self.glu is None else self.glu(x)

        # Conv
        x = self.conv(x)
        # x = self.linear_2(x) # I don't think this makes sense here
        x = self.dropout(x) # F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.conv_layer_norm(x)

        # Linear
        res = x
        if self.norm_before_conv: 
            x = self.final_layer_norm(x)
        x = self.dropout(F.relu(self.fc1(x))) # use gelu?
        x = res + x
        x = self.final_layer_norm(x)
        return x

    
    
    
class Conv1DLSTM_Cov(nn.Module):
    def __init__(self, num_classes=9, kernel_size=5, padding=2):
        super(Conv1DLSTM_Cov, self).__init__()
        self.conv0 = LightweightConv(224, kernel_size, padding)
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
        self.fc = nn.Linear(64, num_classes)
        

    def forward(self, x):
        # input x: batch * 1 * C * T
        y = x.squeeze(1)  # batch * C * T
        y = y.permute(0 ,2, 1)
        y = self.conv0(y)
        y = y.permute(0 ,2, 1)
        y = self.conv1(y)  # batch * 64 * 112
        y = y.transpose(2, 1)  # batch * 112 * 64
        #lstm in: batch, length, feature_in
        #lstm out: batch, length, feature_out * 2(bi)
   

        out, hidden = self.lstm(y)  # out: batch * 112 * 64
        out = out[:, -1, :] # out: batch * 64
        out = self.fc(out)

        return out
    