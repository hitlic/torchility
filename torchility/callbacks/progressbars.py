from sys import stdout
from pytorch_lightning.callbacks import ProgressBarBase


class PrintProgressBar(ProgressBarBase):
    """
    仅输出文本信息的进度条
    """

    def __init__(self, brief=False):
        self.train_batch_id = 0
        self.val_batch_id = 0
        self.test_batch_id = 0
        self.brief = brief    # 是否以简短形式输出信息

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

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.test_batch_id = batch_idx + 1
        stdout.write(f"{self.get_info('test', trainer)}\r")
        stdout.flush()

    def on_test_epoch_end(self, trainer, pl_module):
        print(self.get_info('test', trainer))

    def get_info(self, stage, trainer):
        info = trainer.progress_bar_dict
        if self.brief:
            info_str = ' | '.join([f'{self._brief(k):>12}: {str(v)[:4]:4}' for k, v in info.items() if self._check_info(stage, k)])
        else:
            info_str = ' | '.join([f'{self._brief(k):>12}: {str(v)[:6]:6}' for k, v in info.items() if self._check_info(stage, k)])
        if stage == 'train':
            stage = 'TRAIN'
            c_batch = self.train_batch_id
            num_batch = trainer.num_training_batches
        elif stage == 'val':
            stage = ' VAL '
            c_batch = self.val_batch_id
            num_batch = self._total_val_batches(trainer)
        else:
            stage = ' TST '
            c_batch = self.test_batch_id
            num_batch = sum(trainer.num_test_batches)
        c_epoch = trainer.current_epoch + 1
        max_epoch = trainer.max_epochs

        return f"E:{c_epoch:>3}/{max_epoch:<3}|{stage}| B:{c_batch:>4}/{num_batch:<4}| {info_str}"

    def _total_val_batches(self, trainer):
        total_val_batches = 0
        is_val_epoch = (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch == 0
        total_val_batches = sum(trainer.num_val_batches) if is_val_epoch else 0
        return total_val_batches

    def _brief(self, k):
        if self.brief:
            k_items = k.split('_')
            if len(k_items) == 1:
                return k_items[0][:3]
            x = k_items[0][:3] + '_' + '_'.join([item[0] for item in k_items[1:]])
            return x.replace('step', 's').replace('epoch', 'e')
        else:
            return k.replace('step', 's').replace('epoch', 'e')

    def _check_info(self, stage, info_key):
        if info_key == 'loss' and stage == 'train':
            return True
        if info_key.split('_')[0] != stage:
            return False
        else:
            return True
