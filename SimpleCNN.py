import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.fc1=nn.Linear(32*14*14,128)
        self.fc2=nn.Linear(128,56*56)
        self.ReLU=nn.functional.relu
    def forward(self,x):
        x=self.ReLU(self.conv1(x))
        x=self.pool(x)
        x=self.pool(self.ReLU(self.conv2(x)))
        x=x.view(-1,32*14*14) #展平
        x=self.ReLU(self.fc1(x))
        x=self.fc2(x)
        x=x.view(-1,1,56,56)
        return x