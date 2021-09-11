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
        progress, stage, info = self.get_info(trainer, 'train', 'step')
        print(f"{progress} {stage} {info}", end="\r", flush=True)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        _, train_stage, train_info = self.get_info(trainer, 'train', 'epoch')
        _, val_stage, val_info = self.get_info(trainer, 'val', 'epoch')

        c_epoch = trainer.current_epoch + 1
        max_epoch = trainer.max_epochs
        train_num_batch = trainer.num_training_batches
        val_num_batch = self._total_val_batches(trainer)
        progress = f"E:{c_epoch:>3d}/{max_epoch:<3d} B:{train_num_batch:>4d} {val_num_batch:<4d}"
        print(progress, train_stage, train_info, val_stage, val_info)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_batch_id = batch_idx + 1
        progress, stage, info = self.get_info(trainer, 'val', 'step')
        print(f"{progress} {stage} {info}", end="\r", flush=True)

    def on_test_epoch_start(self, trainer, pl_module):
        print('-' * 23)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.test_batch_id = batch_idx + 1
        progress, stage, info = self.get_info(trainer, 'test', 'step')
        print(f"{progress} {stage} {info}", end="\r", flush=True)

    def on_test_epoch_end(self, trainer, pl_module):
        progress, stage, info = self.get_info(trainer, 'test', 'epoch')
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
        info_str = '  '.join([f'{self._key(k):>}: {v:0<6.4f}' for k, v in info.items() if self._check_info(k, stage, unit)])
        if stage == 'train':
            stage = 'TRAIN >'
            c_batch = self.train_batch_id
            num_batch = trainer.num_training_batches
        elif stage == 'val':
            stage = '  VAL >'
            c_batch = self.val_batch_id
            num_batch = self._total_val_batches(trainer)
        else:
            stage = ' TEST >'
            c_batch = self.test_batch_id
            num_batch = sum(trainer.num_test_batches)
        c_epoch = trainer.current_epoch + 1
        max_epoch = trainer.max_epochs

        return f"E:{c_epoch:>3d}/{max_epoch:<3d} B:{c_batch:>4d}/{num_batch:<4d}", f"| {stage}", f"{info_str}"

    def _total_val_batches(self, trainer):
        total_val_batches = 0
        is_val_epoch = (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch == 0
        total_val_batches = sum(trainer.num_val_batches) if is_val_epoch else 0
        return total_val_batches

    def _key(self, k):
        return k.replace('_step', '').replace('_epoch', '').replace('train_', '').replace('val_', '').replace('test_', '')

    def _check_info(self, info_key, stage, unit):
        key_split = info_key.split('_')
        if key_split[0] != stage or key_split[-1]!=unit:
            return False
        else:
            return True
