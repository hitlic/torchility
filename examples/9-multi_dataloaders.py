import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchility import Trainer
from torchility.utils import set_metric_attr
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
from torchmetrics import Accuracy


# 4. --- 指标和dataloaders
accA = Accuracy(task='multiclass', num_classes=10)
accB = Accuracy(task='multiclass', num_classes=10)

# 设置accA在验证和测试中的第0号dataloader上计算
accA = set_metric_attr(accA, name='accA', dl_idx=0, stages=['val', 'test'])
# 设置accB在验证和测试中的第1号dataloader上计算
accB = set_metric_attr(accB, name='accB', dl_idx=1, stages=['val', 'test'])

val_dls = [val_dl, test_dl]    # [0号，1号]
test_dls = [test_dl, val_dl]   # [0号，1号]


def acc3(preds, targets):
    """同时计算多个指标"""
    preds = preds.argmax(1)
    return {'@a':(preds == targets).float().mean(), '@b':(preds == targets).float().mean()}
acc3 = set_metric_attr(acc3, stages='val', dl_idx=0)

# 5. --- 训练
trainer = Trainer(model, F.cross_entropy, opt, epochs=2, metrics=[accA, accB, acc3])
trainer.fit(train_dl, val_dls)
# trainer.fit(train_dl, val_dls, do_val_loss=[1])    # 训练、验证：仅在1号验证dataloader上计算损失
# trainer.fit(train_dl, val_dls, do_val_loss=1)      # 训练、验证：仅在1号验证dataloader上计算损失

trainer.test(test_dls)
# trainer.test(test_dls, do_loss=1)                  # 测试：在1号测试dataloader上计算损失
# trainer.test(test_dls, do_loss=[0, 1])             # 测试：在0号和1号测试dataloader上计算损失
