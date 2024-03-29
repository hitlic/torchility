import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchility import Trainer
from torchility.callbacks import ModelAnalyzer
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
backward_analyzer = ModelAnalyzer('backward')                       # 对各层反向梯度进行分析
forward_analyzer = ModelAnalyzer('forward')                         # 对各层前向输出进行分析
trainer = Trainer(model, F.cross_entropy, opt, epochs=2,
                  callbacks=[backward_analyzer, forward_analyzer])  # 训练器
trainer.fit(train_dl)


# 5. --- 显示分析结果图像
# 在命令行中运行 tensorboard --logdir=./logs，通过浏览器查看
