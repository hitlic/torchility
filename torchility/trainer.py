from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning import LightningDataModule
import torch
from .tasks import GeneralTaskModule


class Trainer(PLTrainer):
    def compile(self, model: torch.nn.Module, loss, optimizer,
                data_module: LightningDataModule = None,
                log_loss_step=None, log_loss_epoch=True, metrics=None):
        self.task_module = GeneralTaskModule(model, loss, optimizer, log_loss_step, log_loss_epoch, metrics)
        self.data_module = data_module

    def fit(self, train_dl=None, val_dl=None, epochs=10):
        self.max_epochs = epochs
        if self.data_module:
            super().fit(self.task_module, datamodule=self.data_module)
        else:
            super().fit(self.task_module, train_dataloader=train_dl, val_dataloaders=val_dl)

    def test(self, test_dl=None, verbose=True):
        if self.data_module and self.data_module.test_dataloader():
            super().test(self.task_module, datamodule=self.data_module, verbose=verbose)
        else:
            super().test(self.task_module, test_dl, verbose=verbose)
