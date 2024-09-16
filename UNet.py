import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1),  # 3x3卷积，padding=1保持尺寸不变
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(C_out, C_out, kernel_size=3, stride=1, padding=1),  # 再次保持尺寸不变
            nn.BatchNorm2d(C_out),
            nn.Dropout(0.4),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class DownSampling(nn.Module):
    def __init__(self, C_in, C_out):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=2, stride=2),  # 2x2卷积，步幅2会让特征尺寸减半
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.Down(x)

class UpSampling(nn.Module):
    def __init__(self, C_in, C_out):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C_in, C_out, kernel_size=1)  # 改变通道数的卷积

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode='nearest')  # 使用最近邻插值进行上采样
        x = self.Up(up)  # 改变输出通道数
        x = torch.cat([x, r], dim=1)  # 进行跳跃连接，拼接特征
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.C1 = Conv(4, 64)  # 输入图像有4个通道
        self.D1 = DownSampling(64, 128)  # 下采样
        self.C2 = Conv(128, 128)
        # self.D2 = DownSampling(128, 256)  # 第二次下采样
        # self.C3 = Conv(256, 256)
        # self.U1 = UpSampling(256, 128)  # 第一次上采样
        # self.C4 = Conv(256, 128)  # 拼接后通道数为256
        self.U2 = UpSampling(128, 64)  # 第二次上采样
        self.C5 = Conv(128, 64)  # 拼接后通道数为128
        self.pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()  # 对最终输出使用Sigmoid激活函数

    def forward(self, x):
        R1 = self.C1(x)  # 第1层卷积
        R2 = self.C2(self.D1(R1))  # 下采样后卷积
        # R3 = self.C3(self.D2(R2))  # 第二次下采样后卷积

        # up1 = self.U1(R3, R2)  # 第一次上采样，并进行跳跃连接
        up2 = self.U2(R2, R1)  # 第二次上采样，并进行跳跃连接

        c = self.C5(up2)  # 最后一层卷积
        return self.sigmoid(self.pred(c))  # 进行最后的预测并激活




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Conv(nn.Module):
#     def __init__(self, C_in, C_out):
#         super(Conv, self).__init__()
#         self.layer = nn.Sequential(
#         nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1), # 3x3卷积，padding=1保持尺寸不变
#         nn.BatchNorm2d(C_out),
#         nn.Dropout(0.3),
#         nn.LeakyReLU(inplace=True),
#         nn.Conv2d(C_out, C_out, kernel_size=3, stride=1, padding=1), # 再次保持尺寸不变
#         nn.BatchNorm2d(C_out),
#         nn.Dropout(0.4),
#         nn.LeakyReLU(inplace=True),
# )

#     def forward(self, x):
#         return self.layer(x)

# class DownSampling(nn.Module):
#     def __init__(self, C):
#         super(DownSampling, self).__init__()
# # 这里选择了较大的内核和步幅2进行下采样
#         self.Down = nn.Sequential(
#         nn.Conv2d(C, C, kernel_size=2, stride=2), # 4x4卷积，步幅2会让特征尺寸减半
#         nn.LeakyReLU(inplace=True)
# )

#     def forward(self, x):
#         return self.Down(x)

# class UpSampling(nn.Module):
#     def __init__(self, C):
#         super(UpSampling, self).__init__()
#         # 用于1x1卷积调整通道数
#         self.Up = nn.Conv2d(C, C , kernel_size=1) # 改变通道数的卷积

#     # def forward(self, x,r):
#     def forward(self, x):

#         up = F.interpolate(x, scale_factor=2, mode='nearest') # 使用最近邻插值进行上采样
#         x = self.Up(up) # 改变输出通道数
#         return x

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.C1 = Conv(4, 64) # 输入图像有4个通道
#         self.D1 = DownSampling(64) # 下采样
#         self.C2 = Conv(64, 128)
#         self.U1 = UpSampling(128) # 第三层上采样
#         self.C3 = Conv(128, 64)
#         self.pred = nn.Conv2d(64, 1, kernel_size=3, padding=1)
#         self.sigmoid = nn.Sigmoid() # 对最终输出使用Sigmoid激活函数
    
#     def forward(self, x):
#         R1 = self.C1(x) # 第1层卷积
#         R2 = self.C2(self.D1(R1)) # 下采样后卷积
#         c = self.C3(self.U1(R2)) # 上采样后卷积

#         # c = self.C3(self.U1(R2, R1)) # 上采样后卷积
#         return self.sigmoid(self.pred(c)) # 进行最后的预测并激活
