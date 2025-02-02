import torch
from torch import nn
from torch.nn import functional as F


kernel = [[ 0.77966102,  0,          0.77966102],
          [ 0,         -2.11864407,  0       ],
          [ 0.77966102,  0,          0.77966102]]


def kernel_d(w1, w2, k):  # Custom convolution dimension
    return torch.tensor(k, dtype=torch.float32)


class new_conv(nn.Module):   # Rewrite a portion of the convolution using functional
    def __init__(self, in_channels, out_channels):
        super(new_conv, self).__init__()
        # Initialize 3x3 convolutional kernels
        self.kernel = kernel_d(3, 3, kernel)
        # Expand the convolution kernel to the shape of [out_channels, in_channels, 3, 3]
        self.w = nn.Parameter(self.kernel.repeat(out_channels, in_channels, 1, 1))
        self.b = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        out = F.conv2d(x, self.w, self.b, padding=1)
        out = F.relu(out)
        return out


class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
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


class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)

    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)  # N C H W


class UNet(nn.Module):  # UNet
    def __init__(self):
        super(UNet, self).__init__()
        self.c1=Conv_Block(3,64)
        self.c11 = new_conv(64,64)
        self.d1=DownSample(64)
        self.c2=Conv_Block(64,128)
        self.c22 = new_conv(128,128)
        self.d2=DownSample(128)
        self.c3=Conv_Block(128,256)
        self.c33 = new_conv(256,256)
        self.d3=DownSample(256)
        self.c4=Conv_Block(256,512)
        self.c44=new_conv(512,512)
        self.d4=DownSample(512)
        self.c5=Conv_Block(512,1024)
        self.c55=new_conv(1024,1024)
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

    def forward(self,x):
        R1=self.c1(x)
        R11 = self.c11(R1)
        R2=self.c2(self.d1(R11))
        R22 = self.c22(R2)
        R3 = self.c3(self.d2(R22))
        R33 = self.c33(R3)
        R4 = self.c4(self.d3(R33))
        R44 = self.c44(R4)
        R5 = self.c5(self.d4(R44))
        R55 = self.c55(R5)
        O1=self.c6(self.u1(R55,R44))
        O2 = self.c7(self.u2(O1, R33))
        O3 = self.c8(self.u3(O2, R22))
        O4 = self.c9(self.u4(O3, R11))
        return self.Th(self.out(O4))


if __name__ == '__main__':
    x=torch.randn(1,3,256,256)
    net=UNet()
    print(net(x).shape)
