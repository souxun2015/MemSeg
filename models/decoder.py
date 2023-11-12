import torch
import torch.nn as nn


class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConvBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blk(x)


class DBBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DBBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.blk(x)
   

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.upconv3 = UpConvBlock(512, 256)
        self.db3 = DBBlock(512, 256)
        self.upconv2 = UpConvBlock(256, 128)
        self.db2 = DBBlock(256, 128)
        self.upconv1 = UpConvBlock(128, 64)
        self.db1 = DBBlock(128, 64)
        self.upconv0 = UpConvBlock(64, 32)
        self.db0 = DBBlock(96, 48)

        self.upconv00 = UpConvBlock(48, 48)
        self.db00 = DBBlock(48, 24)

        # self.final_conv = nn.Conv2d(48, 2, kernel_size=3, stride=1, padding=1)
        self.final_out = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 2, kernel_size=3, padding=1),
            #nn.Sigmoid(),
        )

    def forward(self, encoder_output, concat_features):
        # concat_features = [level0, level1, level2, level3]
        f0, f1, f2, f3 = concat_features
        # 512 x 8 x 8 -> 512 x 16 x 16
        x_up3 = self.upconv3(encoder_output)
        x_up3 = torch.cat([x_up3, f3], dim=1)
        x_up3 = self.db3(x_up3)  

        # 512 x 16 x 16 -> 256 x 32 x 32
        x_up2 = self.upconv2(x_up3)
        x_up2 = torch.cat([x_up2, f2], dim=1)
        x_up2 = self.db2(x_up2)

        # 256 x 32 x 32 -> 128 x 64 x 64
        x_up1 = self.upconv1(x_up2)
        x_up1 = torch.cat([x_up1, f1], dim=1)  
        x_up1 = self.db1(x_up1)

        # 128 x 64 x 64 -> 96 x 128 x 128
        x_up0 = self.upconv0(x_up1)
        x_up0 = torch.cat([x_up0, f0], dim=1)  
        x_up0 = self.db0(x_up0)

        # 96 x 128 x 128 -> 48 x 256 x 256
        x_up00 = self.upconv00(x_up0)
        x_up00 = self.db00(x_up00)
        
        # 48 x 256 x 256 -> 1 x 256 x 256
        x_mask = self.final_out(x_up00)
        
        
        return x_mask