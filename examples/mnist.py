from torchility import Trainer
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchility.callbacks import PrintProgressBar

import warnings
warnings.filterwarnings('ignore')  # 屏蔽警告信息


# 自定义指标
from torchmetrics import Accuracy as Acc
acc = Acc(num_classes=10)
def accuracy( preds, targets):
    preds = preds.argmax(1)
    return acc(preds, targets)


# 1. --- 数据
data_dir = './dataset'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds= random_split(mnist_full, [55000, 5000])
test_ds = MNIST(data_dir, train=False, transform=transform, download=True)
train_dl = DataLoader(train_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(val_ds, batch_size=32)


# 2. --- 模型
channels, width, height = (1, 28, 28)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(channels * width * height, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 10)
)


# 3. --- 优化器
opt = torch.optim.Adam(model.parameters(), lr=2e-4)


# 4. --- 训练
trainer = Trainer(callbacks=[PrintProgressBar(True)])             # 训练器，可使用默认进度条
trainer.compile(model, F.cross_entropy, opt, metrics=[accuracy])  # 组装
trainer.fit(train_dl, val_dl, 2)                                  # 训练、验证
trainer.test(test_dl)                                             # 测试
