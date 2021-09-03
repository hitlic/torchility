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
        stage, progress, info = self.get_info(trainer, 'train', 'step')
        stdout.write(f"{progress} {stage} {info}\r")
        stdout.flush()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        train_stage, progress, train_info = self.get_info(trainer, 'train', 'epoch')
        val_stage, _, val_info = self.get_info(trainer, 'val', 'epoch')
        print(progress, train_stage, train_info, val_stage, val_info)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_batch_id = batch_idx + 1
        stage, progress, info = self.get_info(trainer, 'val', 'step')
        stdout.write(f"{progress} {stage} {info}\r")
        stdout.flush()

    def on_test_epoch_start(self, trainer, pl_module):
        print('-' * 23)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.test_batch_id = batch_idx + 1
        stage, progress, info = self.get_info(trainer, 'test', 'step')
        stdout.write(f"{progress} {stage} {info}\r")
        stdout.flush()

    def on_test_epoch_end(self, trainer, pl_module):
        stage, progress, info = self.get_info(trainer, 'test', 'epoch')
        print(progress, stage, info)

    def get_info(self, trainer, stage, unit):
        """get training, validation or testing output infomation from Trainer by stage and unit.

        Args:
            trainer (Trainer): Trainer.
            stage (str): 'train', 'val' or 'test'.
            unit (str): 'step' or 'epoch'.

        Returns:
            [str, str, str]: stage str, progress str, information str
        """
        info = trainer.progress_bar_dict
        info_str = '  '.join([f'{self._key(k):>}: {str(v)[:7]:7}' for k, v in info.items() if self._check_info(k, stage, unit)])
        if stage == 'train':
            stage = 'TRAIN >'
            c_batch = self.train_batch_id
            num_batch = trainer.num_training_batches
            loss_str = f'loss: {str(info.get("loss","*"))[:7]:7}'
        elif stage == 'val':
            stage = ' VAL > '
            c_batch = self.val_batch_id
            num_batch = self._total_val_batches(trainer)
            loss_str = f'loss: {str(info.get("val_loss", "*"))[:7]:7}'
        else:
            stage = ' TEST >'
            c_batch = self.test_batch_id
            num_batch = sum(trainer.num_test_batches)
            loss_str = f'loss: {str(info.get("test_loss", "*"))[:7]:7}'
        c_epoch = trainer.current_epoch + 1
        max_epoch = trainer.max_epochs

        return f"| {stage}", f"E:{c_epoch:>3}/{max_epoch:<3} B:{c_batch:>4}/{num_batch:<4}", f"{loss_str}  {info_str}"

    def _total_val_batches(self, trainer):
        total_val_batches = 0
        is_val_epoch = (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch == 0
        total_val_batches = sum(trainer.num_val_batches) if is_val_epoch else 0
        return total_val_batches

    def _key(self, k):
        # return k.replace('step', 's').replace('epoch', 'e').replace('train', 'T').replace('val', 'V').replace('test_', '')
        return k.replace('_step', '').replace('_epoch', '').replace('train_', '').replace('val_', '').replace('test_', '')

    def _check_info(self, info_key, stage, unit):
        if info_key[-4:] == 'loss':
            return False
        key_split = info_key.split('_')
        if key_split[0] != stage or key_split[-1]!=unit:
            return False
        else:
            return True
