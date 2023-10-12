import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
import torch
from torchility import Trainer

import warnings
warnings.simplefilter("ignore")


def fun(x, w):
    return w[0] * x + w[1] * np.power(x, 2)  # w1*x + w2*x^2

def rgression_dataset(fun, noise=True):
    """构造数据集"""
    x = np.linspace(0, 20, 1001)
    w = np.array([-1.5, 1 / 9])
    y = fun(x, w).reshape(-1, 1)
    if noise:
        y += np.random.randn(x.shape[0], 1)
    return x.reshape(-1, 1), y


class ResetableLoader(DataLoader):
    def __init__(self, batch_size=50):
        self.batch_size = batch_size
        x, y = rgression_dataset(fun)  # 生成数据集
        ds = list(zip(x, y))
        super().__init__(ds, batch_size, True, collate_fn=self.collate)

    def reset(self):
        """
        重置数据集
        """
        print('...resetting train dataloader')
        return self.__class__(self.batch_size)

    def collate(self, batch):
        x, y =zip(*batch)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class Model(nn.Module):
    def __init__(self, dim=10):
        super().__init__()
        self.hidden = nn.Linear(1, dim)
        self.out = nn.Linear(dim, 1)
    def forward(self, x):
        y = self.hidden(x)
        y = F.tanh(y)
        return self.out(y)


model = Model()
opt = SGD(model.parameters(), lr=0.001)
train_dl = ResetableLoader()

trainer = Trainer(model, F.mse_loss, opt, 200, reset_dl=5)  # 每5个epoch重置一次dataloader
trainer.fit(train_dl)
