import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchility import Trainer
from torchility.callbacks import ClassifierInterpreter
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")


# 1. --- 数据
data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds = random_split(mnist_full, [55000, 5000])
test_ds = MNIST(data_dir, train=False, transform=transform, download=True)
train_dl = DataLoader(train_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)


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
interpreter = ClassifierInterpreter(class_num=10, stage='test')          # 针对分类模型测试数据的解释器
trainer = Trainer(model, F.cross_entropy, opt,                           # 训练器
                  epochs=2, callbacks=[interpreter]) 
trainer.fit(train_dl, val_dl)                                            # 训练、验证
trainer.test(test_dl)                                                    # 测试


# 5. --- 解释
# 返回测试集中损失最大的10个样本的信息
metric = nn.CrossEntropyLoss(reduction='none')  # 要计算每个样本的指标，因此不能reduction（即求和、平均等操作）
tops = trainer.interpreter.top_samples(metric, k=10)                     # 返回metric值最大的10个样本的信息
print(tops)
trainer.interpreter.plot_confusion()                                     # 绘制混淆矩阵
plt.show()
