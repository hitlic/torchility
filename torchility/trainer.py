from .trainer_base import TrainerBase
from pytorch_lightning import LightningDataModule
import torch
from .tasks import GeneralTaskModule


class Trainer(TrainerBase):
    def compile(self, model: torch.nn.Module, loss, optimizer, data_module: LightningDataModule = None,
                log_loss_step=None, log_loss_epoch=True, metrics=None):
        self.task_module = GeneralTaskModule(model, loss, optimizer, log_loss_step, log_loss_epoch, metrics)
        self.data_module = data_module

    def fit(self, train_dl=None, val_dl=None, epochs=10):
        self.max_epochs = epochs
        if self.data_module:
            super().fit(self.task_module, datamodule=self.data_module)
        else:
            super().fit(self.task_module, train_dataloader=train_dl, val_dataloaders=val_dl)

    def test(self, test_dl=None, ckpt_path='best', pl_module=None, verbose=True):
        if test_dl is not None:
            super().test(pl_module, test_dataloaders=test_dl, ckpt_path=ckpt_path, verbose=verbose)
        elif self.data_module and self.data_module.test_dataloader():
            super().test(pl_module, datamodule=self.data_module, ckpt_path=ckpt_path, verbose=verbose)
        else:
            raise Exception("Dataloader or DataModule is needed!")

    def resume_checkpoint(self, ckpt_path):
        """
        从checkpoint恢复trainer，然后可继续训练。
        注意，再次训练时epochs参数包含已训练的epoches。
        """
        self.init_params['resume_from_checkpoint'] = ckpt_path
        ckpt_trainer = Trainer(**self.init_params)
        ckpt_trainer.task_module = self.task_module
        ckpt_trainer.data_module = self.data_module
        return ckpt_trainer
