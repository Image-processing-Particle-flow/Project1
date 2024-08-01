from DataSet import myDataSet
from easyCNN import easyCNN
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

#超参数设计
BATCH_SIZE=32

#导入数据集
dataset = myDataSet(img_dir='E:/project 2_e', group_size=10000)
transform=transforms.Compose([
    transforms.ToTensor(),
    # 数据预处理后期添加
])

#分割数据集
TRAIN_NUM=int(0.8 * len(dataset))
VAL_NUM=int(0.1 * len(dataset))
TEST_NUM=len(dataset) - TRAIN_NUM - VAL_NUM

train_dataset, val_dataset, test_dataset=random_split(
    dataset,[TRAIN_NUM,VAL_NUM,TEST_NUM]
)

#加载数据迭代器（训练，验证，测试）
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # 验证集不需打乱
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # 测试集不需打乱
# 创立模型
model=easyCNN()
#损失函数和优化器

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs=model(batch_X)
        loss=criterion(outputs,batch_Y)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
    # 在每个 epoch 完成后计算并记录平均损失
    average_loss = running_loss/ len(train_loader) # 计算平均损失
    print(f'Epoch [{epoch+1}/{num_epochs}],Train Loss: {average_loss:.4f}') # 输出平均损失
print('Training finished')
