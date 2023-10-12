from torchility import Trainer
from torchility.utils import metric_config
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score
from torch import nn
import torch
import warnings

warnings.simplefilter("ignore")

# 1. --- 数据
data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
train_ds, val_ds, _ = random_split(mnist_full, [5000, 5000, 50000])
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


# 4. --- 训练
acc0 = Accuracy(task='multiclass', num_classes=10)
acc1 = Accuracy(task='multiclass', average='macro', num_classes=10)

acc0 = metric_config(acc0, name='acc0', stages='val')              # 指定指标名，并指定仅在`验证`中计算
acc1 = metric_config(acc1, name='acc1', stages=['train', 'test'])  # 指定指标在`训练`和`测试`中计算


trainer = Trainer(model, F.cross_entropy, [opt, scheduler], epochs=2,
                  metrics=[acc0, acc1]
                  )
progress = trainer.fit(train_dl, val_dl)            # 训练、验证
trainer.test(test_dl)                               # 测试


# 测试：指定专用于测试的指标
f1 = 'f1', F1Score(task='multiclass', average='macro', num_classes=10)
trainer.test(test_dl, metrics=[f1])
