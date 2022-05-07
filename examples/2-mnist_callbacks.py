import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchility import Trainer
from torchility.callbacks import SimpleBar, Progress
from torchmetrics import Accuracy, F1Score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import time
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
scheduler = torch.optim.lr_scheduler.StepLR(opt, 8, gamma=0.1, last_epoch=-1)


# 4. --- 训练
# 评价指标
acc = 'acc', Accuracy()
acc1 = 'acc1', Accuracy(average='macro', num_classes=10)
f1 = 'f1', F1Score(average='macro', num_classes=10)

# 每个epoch保存一次模型的callback
chkpoint_cbk = ModelCheckpoint(monitor='val_loss', dirpath='./checkpoints/',
                               filename=f'time={int(time.time())}'+'-{epoch:03d}-{val_loss_epoch:.4f}',
                               save_top_k=1, mode='min', every_n_epochs=1)
# 早停callback
early_stop_cbk = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')

# 训练器，使用新的进度条，以及其他callbacks
trainer = Trainer(model, F.cross_entropy, [opt, scheduler], max_epochs=20,
                  metrics=[acc, acc1, f1],
                  callbacks=[SimpleBar(),      # 进度条
                             chkpoint_cbk,     # checkkpoint
                             early_stop_cbk,   # 早停
                             Progress('step')  # 使得fit返回每个batch中的损失和指标
                             ])
progress = trainer.fit(train_dl, val_dl)                               # 训练、验证
trainer.test(test_dl)                                                  # 测试

print(progress)  # 训练过程，包括训练和验证中的损失和其他指标