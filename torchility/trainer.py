from .trainer_base import TrainerBase


class Trainer(TrainerBase):
    def fit(self, train_dl=None, val_dl=None, epochs=None):
        if epochs is not None:
            current_epoch = self.fit_loop.current_epoch
            self.fit_loop.max_epochs = epochs + current_epoch

        if self.datamodule:
            super().fit(self.task_module, datamodule=self.datamodule)
        else:
            super().fit(self.task_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    def test(self, test_dl=None, ckpt_path='best', pl_module=None, verbose=True):
        if test_dl is not None:
            return super().test(pl_module, dataloaders=test_dl, ckpt_path=ckpt_path, verbose=verbose)
        elif self.datamodule and self.datamodule.test_dataloader():
            return super().test(pl_module, datamodule=self.datamodule, ckpt_path=ckpt_path, verbose=verbose)
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
        ckpt_trainer.datamodule = self.datamodule
        return ckpt_trainer
