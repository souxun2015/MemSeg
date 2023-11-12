import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .coordatt import CoordAtt

class MSFFBlock(nn.Module):
    def __init__(self, in_channel):
        super(MSFFBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.act1 = nn.ReLU(inplace=True)

        self.attn = CoordAtt(in_channel, in_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_conv = self.conv1(x)
        x_conv = self.bn1(x_conv)
        x_conv = self.act1(x_conv)

        x_att = self.attn(x)
        x = x_conv * x_att

        x = self.conv2(x)
        return x

    
class MSFF(nn.Module):
    def __init__(self):
        super(MSFF, self).__init__()
        self.blk1 = MSFFBlock(128)
        self.blk2 = MSFFBlock(256)
        self.blk3 = MSFFBlock(512)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.upconv32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.upconv21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # features = [level1, level2, level3]
        f1, f2, f3 = features 
        
        # score map
        m3 = f3[:,256:,...].mean(dim=1, keepdim=True)
        m2 = f2[:,128:,...].mean(dim=1, keepdim=True)
        m1 = f1[:,64:,...].mean(dim=1, keepdim=True)

        # MSFF Module
        f1_k = self.blk1(f1)
        f2_k = self.blk2(f2)
        f3_k = self.blk3(f3)

        f2_f = f2_k + self.upconv32(f3_k)
        f1_f = f1_k + self.upconv21(f2_f)
        
        m1_weight = m1 * self.upsample2(m2) * self.upsample4(m3)
        m2_weight = m2 * self.upsample2(m3)
        m3_weight = m3    

        f1_out = f1_f * m1_weight
        f2_out = f2_f * m2_weight
        f3_out = f3_k * m3_weight
        
        return [f1_out, f2_out, f3_out]