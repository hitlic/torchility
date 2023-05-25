import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class GeneralDataModule(pl.LightningDataModule):
    def __init__(self, train_dl=None, val_dls=None, test_dls=None, pred_dl=None, reset=0):
        super().__init__()
        self.train_dl = train_dl
        self.val_dl = val_dls if val_dls is not None else []
        self.test_dls = test_dls if test_dls is not None else  []
        self.pred_dl = pred_dl
        self.reset = reset > 0

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.reset:
            if hasattr(self.train_dl, 'reset'):
                return self.train_dl.reset()
            else:
                print('\n\033[91mWARNING! Training dataloader should have a `reset` method which returns a NEW dataloader instance!!\0\n')
        return self.train_dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dl

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_dls

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.pred_dl
