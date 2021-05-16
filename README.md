# torchility

A tool for training pytorch deep learning model more simply which is based on Pytorch-lightning.

## Dependency
- pytorch > 1.7
- pytorch-lightning > 1.1

## Usage

- MNIST

```python
from torchility import Trainer
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

# datasets
mnist_full = MNIST(data_dir, train=True, download=True)
train_ds, val_ds = random_split(mnist_full, [55000, 5000])
test_ds = MNIST(data_dir, train=False, download=True)

# dataloaders
train_dl = DataLoader(train_ds, batch_size=32)
val_dl = DataLoader(val_ds, batch_size=32)
test_dl = DataLoader(val_ds, batch_size=32)

# pytorch model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 10)
)

# optimizer
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

# trainer
trainer = Trainer()
# compile
trainer.compile(model, F.cross_entropy, opt)
# train and validate
trainer.fit(train_dl, val_dl, epochs=1)
# test
trainer.test(test_dl)
```

- See the `examples` for more examples 