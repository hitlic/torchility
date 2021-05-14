from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning import LightningDataModule
import torch

from .pl_module import _TorchilityModule


class Trainer(PLTrainer):
    def compile(self, model: torch.nn.Module, loss, optimizer,
                data_module: LightningDataModule = None,
                log_step=True, log_epoch=None):
        self.pl_module = _TorchilityModule(model, loss, optimizer, log_step, log_epoch)
        self.data_module = data_module

    def train(self, train_dl=None, val_dl=None, epochs=10):
        # self.train_loop.trainer
        self.max_epochs = epochs
        if self.data_module:
            super().fit(self.pl_module, datamodule=self.data_module)
        else:
            super().fit(self.pl_module, train_dataloader=train_dl, val_dataloaders=val_dl)

    def test(self, test_dl=None, verbose=True):
        if self.data_module and self.data_module.test_dataloader():
            super().test(self.pl_module, datamodule=self.data_module, verbose=verbose)
        else:
            super().test(self.pl_module, test_dl, verbose=verbose)

    def predict(self, pred_dl):
        if self.data_module and self.data_module.predict_dataloader():
            super().predict(self.pl_module, datamodule=self.data_module)
        else:
            super().predict(self.pl_module, dataloaders=pred_dl)
