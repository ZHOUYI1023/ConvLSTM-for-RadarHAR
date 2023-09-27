import torch
import torch.nn as nn


class depthwise_conv(nn.Module):
    def __init__(self, nin, k =7, kernel_size = 3, stride=1, padding = 1, bias=False):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, k, kernel_size=kernel_size, stride = [stride, 1], padding=padding, groups=nin, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = torch.sum(out,1).unsqueeze(1)
        
        return out
    
    
    
class SepConv(nn.Module):
    def __init__(self, num_classes=9, k = 1, c = 32):
        super(SepConv, self).__init__()
        self.dconv1 = nn.Sequential(
            # input: batch, in_channel, length 
            # conv1d: in_channel, out_channel, kernel, stride, padding
            # size: (in_size-kernel+2*padding)/stride + 1
            # l_out: batch, out_channel, length
            depthwise_conv(1, k, 7, 2),  # out: batch * 32 * 112
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            depthwise_conv(1, k, 5, 1),  # out: batch * 32 * 112
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            depthwise_conv(1, k, 3, 2),  # out: batch * 32 * 112
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True))
        
        self.pconv = nn.Conv1d(218, 32, kernel_size=1)

        self.dconv2 = nn.Sequential(
            # input: batch, in_channel, length 
            # conv1d: in_channel, out_channel, kernel, stride, padding
            # size: (in_size-kernel+2*padding)/stride + 1
            # l_out: batch, out_channel, length
            depthwise_conv(1, k, 7, 2),  # out: batch * 32 * 112
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            depthwise_conv(1, k, 5, 1),  # out: batch * 32 * 112
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            depthwise_conv(1, k, 3, 2),  # out: batch * 32 * 112
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten())
        
        self.fc = nn.Linear(312, num_classes)
        

    def forward(self, x):
        out = self.dconv1(x)
        out = out.squeeze(1).permute(0,2,1)
        out = self.pconv(out)
        out = out.permute(0,2,1).unsqueeze(1)
        out = self.dconv2(out)
        out = self.fc(out)
       
        return out
    