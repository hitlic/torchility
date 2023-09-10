from torchility import Trainer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.nn import functional as F
from torchmetrics import Accuracy
from torch import nn
import torch
import warnings

warnings.simplefilter("ignore")

# 1. --- 数据
data_dir = './datasets'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_full = MNIST(data_dir, train=True, transform=transform, download=True)
# train_ds, val_ds = random_split(mnist_full, [55000, 5000])
train_ds, val_ds, _ = random_split(mnist_full, [5000, 5000, 50000])

test_ds = MNIST(data_dir, train=False, transform=transform, download=True)

'''
DataLoader的每个minibatch数据中，最后一个元素应当为标签。在如下三种情况下需要注意这点：
1. 当定义自己的DataLoader时
   自定义的collate_fn返回的最后一个元素为标签，如果无标签则最后一个元素可为None。
2. 当定义损失函数时
   损失函数包括两个参数，第一个参数为模型forward方法的返回值；
                     第二个函数为DataLoader中minibatch的最后一个元素，即标签。
3. 当定义指标时
   参数与损失函数相同。
'''

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
acc = 'acc', Accuracy(task='multiclass', num_classes=10)
acc1 = 'acc1', Accuracy(task='multiclass', average='macro', num_classes=10)

def acc2(preds, targets):
    preds = preds.argmax(1)
    return (preds == targets).float().mean()

def acc3(preds, targets):
    """同时计算多个指标"""
    preds = preds.argmax(1)
    return {'a':(preds == targets).float().mean(), 'b':(preds == targets).float().mean()}

trainer = Trainer(model, F.cross_entropy, opt, epochs=10,
                  metrics=[acc, acc1, acc2, acc3],    # 指定计算指标，默认在train、val和test中都会计算
                  val_freq=2,                         # 每2个epoch做一次验证
                  )
trainer.fit(train_dl, val_dl)                         # 训练、验证

trainer.test(test_dl)                                 # 测试
