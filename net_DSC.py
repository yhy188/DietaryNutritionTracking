import cv2
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Conv_Block(nn.Module):  # convolution
    def __init__(self,in_channel,out_channel):  # Define Input Dimensions
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.layer(x)

class Conv_BlockDSC(nn.Module):  # DSC
    def __init__(self, in_channel, out_channel):
        super(Conv_BlockDSC, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, groups=in_channel, padding_mode='reflect', bias=False),
            # 3x3的卷积（1）
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

            nn.Conv2d(in_channel, out_channel, 3, 1, 1, groups=in_channel, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()

        )
        self.layer = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self,x):
        return self.layer(x)


class DownSample(nn.Module):  # down sample
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):  # up sample
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)

    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)  # N C H W


class UNet(nn.Module):  # Unet
    def __init__(self):
        super(UNet, self).__init__()
        self.c1=Conv_Block(3,64)
        self.d1=DownSample(64)
        self.c2=Conv_BlockDSC(64,128)
        self.d2=DownSample(128)
        self.c3=Conv_BlockDSC(128,256)
        self.d3=DownSample(256)
        self.c4=Conv_BlockDSC(256,512)
        self.d4=DownSample(512)
        self.c5=Conv_BlockDSC(512,1024)
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out=nn.Conv2d(64,3,3,1,1)
        self.Th=nn.Sigmoid()

    def forward(self,x):  # forward
        # print("x",x.shape)
        R1=self.c1(x)
        # print("R1",R1.shape)
        R2=self.c2(self.d1(R1))
        # print("R2", R2.shape)
        R3 = self.c3(self.d2(R2))
        # print("R3", R3.shape)
        R4 = self.c4(self.d3(R3))
        # print("R4", R4.shape)
        R5 = self.c5(self.d4(R4))
        # print("R5", R5.shape)
        O1=self.c6(self.u1(R5,R4))  # contact
        # print("----o1", O1.shape, "5", R5.shape, "4", R4.shape)
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))


if __name__ == '__main__':
    x=torch.randn(1,3,256,256)  # test
    net=UNet()
    print(net(x).shape)
    for name, parameters in net.named_parameters():
       print(name, ':', parameters.size())
