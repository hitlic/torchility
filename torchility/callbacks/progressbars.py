from sys import stdout
from pytorch_lightning.callbacks import ProgressBarBase


class PrintProgressBar(ProgressBarBase):
    """
    仅输出文本信息的进度条
    """

    def __init__(self):
        self.train_batch_id = 0
        self.val_batch_id = 0
        self.test_batch_id = 0

    def disable(self):
        self.enable = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.train_batch_id = batch_idx + 1
        stdout.write(f"{self.get_info('train', trainer)}\r")
        stdout.flush()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        print(self.get_info('train', trainer))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_batch_id = batch_idx + 1
        stdout.write(f"{self.get_info('val', trainer)}\r")
        stdout.flush()

    def on_validation_epoch_end(self, trainer, pl_module):
        print(self.get_info('val', trainer))

    def on_test_epoch_start(self, trainer, pl_module):
        print('-' * 10)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.test_batch_id = batch_idx + 1
        stdout.write(f"{self.get_info('test', trainer)}\r")
        stdout.flush()

    def on_test_epoch_end(self, trainer, pl_module):
        print(self.get_info('test', trainer))

    def get_info(self, stage, trainer):
        info = trainer.progress_bar_dict
        info_str = ' | '.join([f'{self._key(k):>8}: {str(v)[:7]:7}' for k, v in info.items() if self._check_info(stage, k)])
        if stage == 'train':
            stage = 'TRAIN'
            c_batch = self.train_batch_id
            num_batch = trainer.num_training_batches
            loss_str = f'T_loss: {str(info.get("loss","nan"))[:6]:6}'
        elif stage == 'val':
            stage = ' VAL '
            c_batch = self.val_batch_id
            num_batch = self._total_val_batches(trainer)
            loss_str = f'V_loss: {str(info.get("val_loss", "nan"))[:6]:6}'
        else:
            stage = ' TST '
            c_batch = self.test_batch_id
            num_batch = sum(trainer.num_test_batches)
            loss_str = f'  loss: {str(info.get("test_loss", "nan"))[:6]:6}'
        c_epoch = trainer.current_epoch + 1
        max_epoch = trainer.max_epochs

        return f"E:{c_epoch:>3}/{max_epoch:<3}|{stage}| B:{c_batch:>4}/{num_batch:<4} | {loss_str} | {info_str}"

    def _total_val_batches(self, trainer):
        total_val_batches = 0
        is_val_epoch = (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch == 0
        total_val_batches = sum(trainer.num_val_batches) if is_val_epoch else 0
        return total_val_batches

    def _key(self, k):
        return k.replace('step', 's').replace('epoch', 'e').replace('train', 'T').replace('val', 'V').replace('test_', '')

    def _check_info(self, stage, info_key):
        if info_key[-4:] == 'loss':
            return False
        if info_key.split('_')[0] != stage:
            return False
        else:
            return True
