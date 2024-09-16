import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义自注意力层
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1) #H*W->H*W
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        # 生成查询、键、值
        queries = self.query(x).view(batch_size, C, -1) # (B, C, H*W)
        keys = self.key(x).view(batch_size, C, -1) # (B, C, H*W)
        values = self.value(x).view(batch_size, C, -1) # (B, C, H*W)

        # 计算自注意力
        attention_scores = torch.bmm(queries.permute(0, 2, 1), keys) # (B, H*W, H*W)
        attention_scores = self.softmax(attention_scores)

        out = torch.bmm(values, attention_scores.permute(0, 2, 1)) # (B, C, H*W)
        return out.view(batch_size, C, H, W) #不改变形状

# 定义U-Net与Transformer结合的模型
class UNetTransformer(nn.Module):
    def __init__(self, in_channels, out_channels=1): # 输出通道调整为1
        super(UNetTransformer, self).__init__()
        # 编码器部分
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1) # 输入通道为4 
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # 自注意力层
        self.attention = SelfAttention(256) #输入256个通道进入
        # 解码器部分
        self.dec1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv1=nn.Conv2d(256,128,kernel_size=1)
        self.dec2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1) # 最终输出通道为1
        self.conv2=nn.Conv2d(128,64,kernel_size=1)
    def forward(self, x):
        # 编码
        enc1 = F.relu(self.enc1(x)) # 4*56*56->64*56*56
        enc2 = F.relu(self.enc2(F.max_pool2d(enc1, 2))) #64*28*28->128*28*28
        enc3 = F.relu(self.enc3(F.max_pool2d(enc2, 2))) #128*14*14->256*14*14

        # 自注意力
        attn_out = self.attention(enc3) # 256*14*14->256*14*14
       
        # 解码
        dec1=F.relu(self.dec1(F.upsample(attn_out,scale_factor=2,mode='bilinear',align_corners=False))) #128*28*28
        dec1=torch.cat([dec1,enc2],dim=1) # 256*28*28
        dec1=self.conv1(dec1)
        dec2 = F.relu(self.dec2(F.upsample(dec1, scale_factor=2, mode='bilinear', align_corners=False))) #128*28*28->64*56*56
        dec2 = torch.cat([dec2, enc1], dim=1) # 跳跃连接
        dec2=self.conv2(dec2) #B 64 56 56
        out = self.dec3(dec2) #64*56*56->1*56*56
        
        return out