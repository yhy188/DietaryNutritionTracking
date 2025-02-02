import torch
from torch import nn
from torch.nn import functional as F
import random
import time


class Conv_Block(nn.Module):  # DSC
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, groups=in_channel, padding_mode='reflect', bias=False),
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

    def forward(self, x):
        return self.layer(x)


class Conv_Block_up(nn.Module):
    def __init__(self, in_channel, out_channel):  # Upsampling convolution
        super(Conv_Block_up, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class Conv_Block_one(nn.Module):
    def __init__(self, in_channel, out_channel):  # random convolution
        super(Conv_Block_one, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        start = time.perf_counter()
        matrix_4D = self.layer(x)  # The convolved x matrix contains tensor information of gradients

        matrix_3D = matrix_4D[0]
        size = matrix_3D.shape

        C = size[0]
        W = size[1]
        H = size[2]
        matrix_4D = torch.reshape(matrix_4D,(1, C, 1, H * W))  # Matrix vectorization
        # print(matrix_4D.shape)

        # rate_C = 0.1
        rate = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Length of 10, random transformation
        sample = random.choices(rate, k=H * W * C)
        sample = torch.tensor(sample)
        sample = torch.reshape(sample, (1, C, 1, H * W))
        device = torch.device('cuda')
        sample = sample.to(device)  # Put data on GPU, debug without GPU
        matrix_4D = torch.mul(matrix_4D, sample)
        matrix_4D = torch.reshape(matrix_4D, (1, C, H, W))
        end = time.process_time()
        print("time:",end-start)
        return matrix_4D


class DownSample(nn.Module):  # Define downsampling
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):  # up sample
    def __init__(self, channel):
        super(UpSample, self).__init__()
        # print("channel",channel)
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


 # Standard deviation: 1,size=9*9
kernel = [[0,          0,          0.00015924,  0.00047771,  0.00079618,  0.00047771, 0.00015924,  0,          0],
          [0,          0.00031847,  0.00286624,  0.00971338,  0.01417197,  0.00971338, 0.00286624,  0.00031847,  0        ],
          [0.00015924,  0.00286624,  0.02038217,  0.0522293,   0.06464968,  0.0522293, 0.02038217,  0.00286624,  0.00015924],
          [0.00047771,  0.00971338,  0.0522293,   0.05859873,  0,          0.05859873, 0.0522293,   0.00971338,  0.00047771],
          [0.00079618,  0.01417197,  0.06464968,  0,         -0.15923567,  0, 0.06464968,  0.01417197,  0.00079618],
          [0.00047771,  0.00971338,  0.0522293,   0.05859873,  0,          0.05859873, 0.0522293,   0.00971338,  0.00047771],
          [0.00015924,  0.00286624,  0.02038217,  0.0522293,   0.06464968,  0.0522293, 0.02038217,  0.00286624,  0.00015924],
          [ 0,          0.00031847,  0.00286624,  0.00971338,  0.01417197,  0.00971338, 0.00286624,  0.00031847,  0        ],
          [ 0,         0,          0.00015924,  0.00047771,  0.00079618,  0.00047771, 0.00015924,  0,          0        ]
          ]


def kernel_d(w1,w2,k):  # Custom convolution dimension
    k0 = []
    k00 = []
    for i in range(w1):
        k0.append(k)
    for j in range(w2):
        k00.append(k0)
    k00 = torch.tensor(k00)
    return k00


class new_conv(nn.Module):   # Rewrite a portion of the convolution using functional
    def __init__(self,w_1,w_2,w_3,w_4,b,kernel):
        super(new_conv,self).__init__()
        self.kernel = kernel_d(w_1,w_2,kernel)
        self.w = nn.Parameter(self.kernel)   # Customize weights and biases
        self.b = nn.Parameter(torch.randn(b))

    def forward(self,x):
        out = F.conv2d(x,self.w,self.b,padding=4)
        out = F.relu(out)
        return out


class UNet(nn.Module):  # RDL-Unet
    def __init__(self):
        super(UNet, self).__init__()
        self.c1 = Conv_Block_one(3, 64)  # Random convolution
        self.c11 = new_conv(64,64,3,3,64,kernel)  # Enhanced filtering convolution
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.c22 = new_conv(128, 128, 3, 3, 128, kernel)  # Enhanced filtering convolution
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.c33 = new_conv(256, 256, 3, 3, 256, kernel)  # Enhanced filtering convolution
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.c44 = new_conv(512, 512, 3, 3, 512, kernel)  # Enhanced filtering convolution
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.c55 = new_conv(1024, 1024, 3, 3, 1024, kernel)  # Enhanced filtering convolution
        self.u1 = UpSample(1024)  # 上采样，输入1024

        self.c6 = Conv_Block_up(1024, 512)
        self.u2 = UpSample(512)  # 上采样，输入512
        self.c7 = Conv_Block_up(512, 256)
        self.u3 = UpSample(256)  # 上采样 输入256
        self.c8 = Conv_Block_up(256, 128)
        self.u4 = UpSample(128)  # 上采样，输入128
        self.c9 = Conv_Block_up(128, 64)
        self.out = nn.Conv2d(64, 3, 3, 1, 1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        R1 = self.c1(x)
        R11 = self.c11(R1)
        R2 = self.c2(self.d1(R11))
        R22 = self.c22(R2)
        R3 = self.c3(self.d2(R22))
        R33 = self.c33(R3)
        R4 = self.c4(self.d3(R33))
        R44 = self.c44(R4)
        R5 = self.c5(self.d4(R44))
        R55 = self.c55(R5)
        O1 = self.c6(self.u1(R55, R44))
        O2 = self.c7(self.u2(O1, R33))
        O3 = self.c8(self.u3(O2, R22))
        O4 = self.c9(self.u4(O3, R11))
        return self.Th(self.out(O4))

# Note: When testing the model structure, it is necessary to adjust the data in the Conv_Slock_one
# function to not be on the GPU


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    net = UNet()
    print("----", net(x).shape)

