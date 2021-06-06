from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data.dataset import random_split
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
        if test_dl is not None:
            super().test(self.task_module, test_dl, verbose=verbose)
        elif self.data_module and self.data_module.test_dataloader():
            super().test(self.task_module, datamodule=self.data_module, verbose=verbose)
        else:
            raise Exception("Dataloader or DataModule is needed!")

    def load_checkpoint(self, ckp_path, **ckp_args):
        task_module_class = self.task_module.__class__
        pl_model = task_module_class.load_from_checkpoint(ckp_path,
                                                          model=self.task_module.model,
                                                          loss=self.task_module.loss_fn,
                                                          optimizer=self.task_module.opt,
                                                          **ckp_args)
        self.task_module = pl_model

    def test_checkpoint(self, test_dl=None, ckp_path=None, verbose=True, **ckp_args):
        """
        利用checkpoint进行测试， ckp is short for checkpoint。
        如果没有提供ckp_path，则去callback中查找ModelCheckpoint中的best_model_path。
        """
        if ckp_path is None:
            model_checkpoint = None
            for cbk in self.callbacks:
                if isinstance(cbk, ModelCheckpoint):
                    model_checkpoint = cbk
                    break
            if model_checkpoint.best_model_path:
                ckp_path = model_checkpoint.best_model_path
        if ckp_path is not None:
            self.load_checkpoint(ckp_path, **ckp_args)
        else:
            raise Exception("No checkpoint!")
        self.test(test_dl, verbose)
