from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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


# 4. --- 训练

# 早停callback
early_stop_cbk = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')
# 模型checkpoint callback
model_cbk = ModelCheckpoint(save_top_k=1, monitor='val_loss', mode='min')

# 训练器，使用新的进度条，以及其他callbacks
trainer = Trainer(model, F.cross_entropy, opt, epochs=2,
                  callbacks=[
                                model_cbk,
                                early_stop_cbk
                            ])
trainer.fit(train_dl, val_dl)

trainer.test(test_dl)
