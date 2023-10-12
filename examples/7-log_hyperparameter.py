from torchility import Trainer
from torchility.utils import metric_config
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
train_ds, val_ds, _ = random_split(mnist_full, [5000, 5000, 50000])
test_ds = MNIST(data_dir, train=False, transform=transform, download=True)
train_dl = DataLoader(train_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(test_ds, batch_size=32)

dropout = 0.1
dim = 64
lr = 0.01
from itertools import product

lr_s = [0.001, 0.01]
dim_s = [32, 64]
dropout_s = [0.1, 0.5]

for lr, dim, dropout in product(lr_s, dim_s, dropout_s):
    # 2. --- 模型
    channels, width, height = (1, 28, 28)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(channels * width * height, dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim, 10)
    )


    # 3. --- 优化器
    opt = torch.optim.SGD(model.parameters(), lr=lr)


    # 4. --- 训练
    acc0 = metric_config(Accuracy(task='multiclass', num_classes=10), name='acc0')
    acc1 = metric_config(Accuracy(task='multiclass', average='macro', num_classes=10), name='acc1')


    trainer = Trainer(model, F.cross_entropy, opt, epochs=10,
                    metrics=[acc0, acc1],
                    hyper_parameters={'lr': lr, 'dim': dim, 'dropout': dropout},  # 将超参数记入日志
                    )
    progress = trainer.fit(train_dl, val_dl)            # 训练、验证
    trainer.test(test_dl)                               # 测试

# 运行tensorboard，如果使用其他日志工具，则请手动启动相应工具查看日志
#     手动运行tensorboard：
#        在命令行终端输入 tensorboard --logdir path/of/log
trainer.run_tensorboard()
