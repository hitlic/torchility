from torchility import Trainer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.nn import functional as F
from torch import nn
import torch
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
scheduler = torch.optim.lr_scheduler.StepLR(opt, 8, gamma=0.1, last_epoch=-1)

opt_tuple = (opt, scheduler)

opt_dict = {
        'optimizer': opt,
        'lr_scheduler': scheduler
      }

# 4. --- 训练
trainer = Trainer(model, F.cross_entropy, opt_tuple, epochs=10)
# 或者
# trainer = Trainer(model, F.cross_entropy, opt_dict, epochs=10)
trainer.fit(train_dl, val_dl)               # 训练、验证


trainer.test(test_dl)                       # 测试
