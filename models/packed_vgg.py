import torch
from torch import nn
from einops import rearrange
from .packed_layers import PackedConv2d, PackedLinear


# 7-layer VGG

class PackedVGG(nn.Module):
    def __init__(self, n_classes):
        super(PackedVGG, self).__init__()
        M = 4
        alpha = 2
        gamma = 1
        self.features = nn.Sequential(
            PackedConv2d(3, 32, (3, 3), alpha=alpha, num_estimators=M, gamma=gamma, first=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PackedConv2d(32, 32, (3, 3), alpha=alpha, num_estimators=M, gamma=gamma),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            PackedConv2d(32, 32, (3, 3), alpha=alpha, num_estimators=M, gamma=gamma),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            PackedConv2d(32, 64, (3, 3), alpha=alpha, num_estimators=M, gamma=gamma),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PackedConv2d(64, 64, (3, 3), alpha=alpha, num_estimators=M, gamma=gamma),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            PackedConv2d(64, 128, (3, 3), alpha=alpha, num_estimators=M, gamma=gamma),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            PackedConv2d(128, 128, (3, 3), alpha=alpha, num_estimators=M, gamma=gamma),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = PackedLinear(256, M*n_classes, alpha=alpha, num_estimators=M, gamma=gamma, last=True)
        self.num_estimators = M


    def forward(self, x):
        x = self.features(x)
        x = rearrange(x, "e (m c) h w -> (m e) c h w", m=self.num_estimators)
        avg_ = self.avg_pool(x)
        max_ = self.max_pool(x)
        fc_in = torch.cat([avg_, max_], dim=1)
        fc_in = torch.flatten(fc_in, 1)
        fc_in = self.fc(fc_in)
        return fc_in