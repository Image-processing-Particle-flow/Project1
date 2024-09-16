import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cov3x3 = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.cov5x5 = nn.Conv2d(4, 2, kernel_size=5, padding=2)
        self.cov7x7 = nn.Conv2d(4, 2, kernel_size=7, padding=3)

        self.encoder=nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # self.se1=SEBlock(128)
        self.decoder=nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
    def forward(self,x): #x=torch.cat((emcal,hcal,trkn,trkp),dim=1) (4,56,56)
        x1=self.cov3x3(x)
        x2=self.cov5x5(x)
        x3=self.cov7x7(x)
        x = torch.cat((x1,x2,x3),dim=1)
        x=self.encoder(x)
        x=self.decoder(x)
        return x