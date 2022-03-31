from pytorch_lightning.callbacks import ProgressBarBase
from pytorch_lightning.callbacks.base import Callback
from itertools import chain

class ProgressMix:
    """
    获取训练过程信息的混入类。

    实现类至少需要包含如下4个方法，以得到当前的batch序号：
        def __init__(self):
            self.train_batch_id = 0
            self.val_batch_id = 0
            self.test_batch_id = 0
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            self.train_batch_id = batch_idx + 1
        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            self.val_batch_id = batch_idx + 1
        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            self.test_batch_id = batch_idx + 1
    """

    def get_info(self, trainer, stage, unit):
        """get training, validation or testing output infomation from Trainer by stage and unit.

        Args:
            trainer (Trainer): Trainer.
            stage (str): 'train', 'val' or 'test'.
            unit (str): 'step' or 'epoch'.
        Returns:
            [int, int, int, int, dict]: current epoch, total epochs, current batch, total batchs, information dict
        """
        info = trainer.progress_bar_callback.get_metrics(trainer, trainer.task_module)
        info_dict = {self._key(k): v for k, v in info.items() if self._check_info(k, stage, unit)}
        if stage == 'train':
            c_batch = self.train_batch_id
            num_batch = trainer.num_training_batches
        elif stage == 'val':
            c_batch = self.val_batch_id
            num_batch = self._total_val_batches(trainer)
        else:
            c_batch = self.test_batch_id
            num_batch = sum(trainer.num_test_batches)
        c_epoch = trainer.current_epoch + 1
        num_epoch = trainer.max_epochs

        return c_epoch, num_epoch, c_batch, num_batch, info_dict

    def _total_val_batches(self, trainer):
        total_val_batches = 0
        is_val_epoch = (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch == 0
        total_val_batches = sum(trainer.num_val_batches) if is_val_epoch else 0
        return total_val_batches

    def _key(self, k):
        return k.replace('_step', '').replace('_epoch', '').replace('train_', '').replace('val_', '').replace('test_', '')

    def _check_info(self, info_key, stage, unit):
        key_split = info_key.split('_')
        if key_split[0] != stage or key_split[-1] != unit:
            return False
        else:
            return True


class SimpleBar(ProgressBarBase, ProgressMix):
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
        c_epoch, num_epoch, c_batch, num_batch, info_dict = self.get_info(trainer, 'train', 'step')
        progress = f"E:{c_epoch:>3d}/{num_epoch:<3d} B:{c_batch:>4d}/{num_batch:<4}"
        stage = 'TRAIN >'
        info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in info_dict.items()])
        print(f"{progress} {stage} {info}", end="\r", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        train_c_epoch, train_num_epoch, train_c_batch, train_num_batch, train_info_dict = self.get_info(
            trainer, 'train', 'epoch')
        val_c_epoch, val_num_epoch, val_c_batch, val_num_batch, val_info_dict = self.get_info(trainer, 'val', 'epoch')

        progress = f"E:{train_c_epoch:>3d}/{train_num_epoch:<3d} B:{train_num_batch:>4} {val_num_batch:<4}"
        train_stage = 'TRAIN >'
        train_info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in train_info_dict.items()])
        val_stage = '  VAL >'
        val_info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in val_info_dict.items()])
        print(progress, train_stage, train_info, val_stage, val_info)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_batch_id = batch_idx + 1
        c_epoch, num_epoch, c_batch, num_batch, info_dict = self.get_info(trainer, 'val', 'step')
        progress = f"E:{c_epoch:>3d}/{num_epoch:<3d} B:{c_batch:>4d}/{num_batch:<4}"
        stage = '  VAL >'
        info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in info_dict.items()])
        print(f"{progress} {stage} {info}", end="\r", flush=True)

    def on_test_epoch_start(self, trainer, pl_module):
        print('-' * 23)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.test_batch_id = batch_idx + 1
        c_epoch, num_epoch, c_batch, num_batch, info_dict = self.get_info(trainer, 'test', 'step')
        progress = f"E:{c_epoch:>3d}/{num_epoch:<3d} B:{c_batch:>4d}/{num_batch:<4}"
        stage = ' TEST >'
        info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in info_dict.items()])
        print(f"{progress} {stage} {info}", end="\r", flush=True)

    def on_test_epoch_end(self, trainer, pl_module):
        c_epoch, num_epoch, c_batch, num_batch, info_dict = self.get_info(trainer, 'test', 'epoch')
        progress = f"E:{c_epoch:>3d}/{num_epoch:<3d} B:{c_batch:>4d}/{num_batch:<4}"
        stage = ' TEST >'
        info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in info_dict.items()])
        print(progress, stage, info)


PrintProgressBar = SimpleBar


class Progress(Callback, ProgressMix):
    """
    把获取信息之类的工具转移到该类中，令PrintProgressBar作为该类的子类。将PrintProgressBar改名为SimpleBar。
    """

    def __init__(self, unit='epoch'):
        assert unit in ['epoch', 'step'], '`unit` must be epoch or step'
        self.unit = unit

        self.train_batch_id = 0
        self.val_batch_id = 0
        self.test_batch_id = 0

        self.progress_info = []
        self.curent_epoch = None
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        if self.unit == 'epoch':
            self.curent_epoch = {'train_epoch': None, 'val_epoch': None}
        else:
            self.curent_epoch = {'train_epoch': [], 'val_epoch': []}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.train_batch_id = batch_idx + 1
        if self.unit == 'step':
            _, _, _, _, info_dict = self.get_info(trainer, 'train', 'step')
            self.curent_epoch['train_epoch'].append(info_dict)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_batch_id = batch_idx + 1
        if self.unit == 'step':
            _, _, _, _, info_dict = self.get_info(trainer, 'val', 'step')
            self.curent_epoch['val_epoch'].append(info_dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.unit == 'epoch':
            _, _, _, _, info_dict = self.get_info(trainer, 'val', 'epoch')
            self.curent_epoch['val_epoch'] = info_dict

    def on_train_epoch_end(self, trainer, pl_module):
        if self.unit == 'epoch':
            _, _, _, _, info_dict = self.get_info(trainer, 'train', 'epoch')
            self.curent_epoch['train_epoch'] = info_dict
        self.progress_info.append(self.curent_epoch)

    def on_train_end(self, trainer, pl_module):
        if self.unit == 'epoch':
            metrics = self.progress_info[0]['train_epoch'].keys()
            train_info = {
                metric: [e['train_epoch'][metric] for e in self.progress_info]
                for metric in metrics
            }
            val_info = {
                metric: [e['val_epoch'][metric] for e in self.progress_info if e['val_epoch']]
                for metric in metrics
            }
            trainer.progress = {'train': train_info, 'val': val_info}
        else:
            metrics = self.progress_info[0]['train_epoch'][0].keys()
            train_info = {
                metric: list(chain.from_iterable([[b[metric] for b in e['train_epoch']] for e in self.progress_info]))
                for metric in metrics
            }
            val_info = {
                metric: list(chain.from_iterable([[b[metric] for b in e['val_epoch']] for e in self.progress_info if e['val_epoch']]))
                for metric in metrics
            }
            trainer.progress = {'train': train_info, 'val': val_info}