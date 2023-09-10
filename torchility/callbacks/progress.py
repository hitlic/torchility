from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.callbacks.callback import Callback
from itertools import chain
import pytorch_lightning as pl


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
    def get_info(self, trainer, stage):
        """
        返回当前epoch数，总epoch数，当前batch总，总batch数。
        Args:
            trainer (Trainer): Trainer.
            stage (str): 'train', 'val' or 'test'.
        Returns:
            [int, int, int, int]: current epoch, total epochs, current batch, total batchs
        """
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

        return c_epoch, num_epoch, c_batch, num_batch

    def _total_val_batches(self, trainer):
        total_val_batches = 0
        is_val_epoch = (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch == 0
        total_val_batches = sum(trainer.num_val_batches) if is_val_epoch else 0
        return total_val_batches

    def get_epoch_loss(self, trainer, stage):
        cbk_metrics = trainer.callback_metrics
        if stage == 'train':
            return {'loss': cbk_metrics['train_loss_epoch'].item()}
        else:
            return {self._loss_key(k): v.item() for k,v in cbk_metrics.items() if f'{stage}_loss_epoch' in k}

    def _loss_key(self, k):
        if '/dataloader_idx_' in k:
            return f"loss/{k.split('_')[-1]}"
        elif k in ['val_loss_epoch', 'test_loss_epoch']:
            return 'loss'
        else:
            return k


class SimpleBar(ProgressBar, ProgressMix):
    """
    自定义进度显示
    """
    def __init__(self, long_output=False):
        self.train_batch_id = 0
        self.val_batch_id = 0
        self.test_batch_id = 0
        self.has_val = False
        self.long_output = long_output


    def disable(self):
        self.enable = True

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int):
        self.train_batch_id = batch_idx + 1
        c_epoch, num_epoch, c_batch, num_batch = self.get_info(trainer, 'train')
        progress = f"E:{c_epoch:>3d}/{num_epoch:<3d} B:{c_batch:>4d}/{num_batch:<4}"
        stage = 'TRAIN >'
        if self.long_output:
            info = '  '.join([f'{k:>}: {v:0<9.7f}' for k, v in outputs.items()])
        else:
            info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in outputs.items()])
        print(f"\r{progress} {stage} {info}", end="", flush=True)
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_epoch_end(self, trainer, pl_module):
        train_c_epoch, train_num_epoch, _, train_num_batch = self.get_info(trainer, 'train')
        _, _, _, val_num_batch = self.get_info(trainer, 'val')

        train_info_dict = self.get_epoch_loss(trainer, 'train')
        train_info_dict.update(trainer.train_epoch_metric_values)

        val_info_dict = {}
        if len(trainer.datamodule.val_dataloader()) > 0 and self.has_val:
            val_info_dict = self.get_epoch_loss(trainer, 'val')
            val_info_dict.update(trainer.val_epoch_metric_values)

        progress = f"\rE:{train_c_epoch:>3d}/{train_num_epoch:<3d} B:{train_num_batch:>4} {val_num_batch:<4}"
        train_stage = 'TRAIN >'
        if self.long_output:
            train_info = '  '.join([f'{k:>}: {v:0<9.7f}' for k, v in train_info_dict.items()])
        else:
            train_info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in train_info_dict.items()])

        val_stage = '  VAL >'
        if self.long_output:
            val_info = '  '.join([f'{k:>}: {v:0<9.7f}' for k, v in val_info_dict.items()])
        else:
            val_info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in val_info_dict.items()])
        if not val_info:
            val_stage = ''
        print(progress, train_stage, train_info, val_stage, val_info)
        self.has_val = False

    def on_validation_epoch_end(self, trainer, pl_module):
        self.has_val = True  # 关前训练epoch是否包含验证

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        self.val_batch_id = batch_idx + 1
        c_epoch, num_epoch, c_batch, num_batch  = self.get_info(trainer, 'val')
        progress = f"E:{c_epoch:>3d}/{num_epoch:<3d} B:{c_batch:>4d}/{num_batch:<4}"
        stage = '  VAL >'
        if self.long_output:
            info = '  '.join([f'{k:>}: {v:0<9.7f}' for k, v in outputs.items()])
        else:
            info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in outputs.items()])
        print(f"\r{progress} {stage} {info}", end="", flush=True)
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0):
        self.test_batch_id = batch_idx + 1
        c_epoch, num_epoch, c_batch, num_batch = self.get_info(trainer, 'test')
        progress = f"E:{c_epoch:>3d}/{num_epoch:<3d} B:{c_batch:>4d}/{num_batch:<4}"
        stage = ' TEST >'
        if self.long_output:
            info = '  '.join([f'{k:>}: {v:0<9.7f}' for k, v in outputs.items()])
        else:
            info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in outputs.items()])
        print(f"\r{progress} {stage} {info}", end="", flush=True)
        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self, trainer, pl_module):
        c_epoch, num_epoch, c_batch, num_batch = self.get_info(trainer, 'test')
        info_dict = self.get_epoch_loss(trainer, 'test')
        info_dict.update(trainer.test_epoch_metric_values)
        progress = f"E:{c_epoch:>3d}/{num_epoch:<3d} B:{c_batch:>4d}/{num_batch:<4}"
        stage = ' TEST >'
        if self.long_output:
            info = '  '.join([f'{k:>}: {v:0<9.7f}' for k, v in info_dict.items()])
        else:
            info = '  '.join([f'{k:>}: {v:0<6.4f}' for k, v in info_dict.items()])
        trainer.test_metrics = info_dict
        print(f"\r{progress} {stage} {info}")


PrintProgressBar = SimpleBar


class Progress(Callback, ProgressMix):
    """
    保存训练、验证过程中的指标数据，用于作为 trainer.fit 方法的返回结果
    """
    def __init__(self, unit='epoch'):
        assert unit in ['epoch', 'step'], '`unit` must be epoch or step'
        self.unit = unit

        self.train_batch_id = 0
        self.val_batch_id = 0
        self.test_batch_id = 0

        self.progress_info = []
        self.curent_epoch = {}
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        if self.unit == 'epoch':
            self.curent_epoch = {'train_epoch': {}, 'val_epoch': {}}
        else:
            self.curent_epoch = {'train_epoch': [], 'val_epoch': []}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_batch_id = batch_idx + 1
        if self.unit == 'step':
            self.curent_epoch['train_epoch'].append(outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_batch_id = batch_idx + 1
        if self.unit == 'step':
            self.curent_epoch['val_epoch'].append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.unit == 'epoch':
            info_dict = self.get_epoch_loss(trainer, 'val')
            info_dict.update(trainer.val_epoch_metric_values)
            self.curent_epoch['val_epoch'] = info_dict

    def on_train_epoch_end(self, trainer, pl_module):
        if self.unit == 'epoch':
            info_dict = self.get_epoch_loss(trainer, 'train')
            info_dict.update(trainer.train_epoch_metric_values)
            self.curent_epoch['train_epoch'] = info_dict
        self.progress_info.append(self.curent_epoch)

    def on_test_epoch_end(self, trainer, pl_module):
        if self.unit == 'epoch':
            info_dict = self.get_epoch_loss(trainer, 'test')
            info_dict.update(trainer.test_epoch_metric_values)
            self.curent_epoch['test_epoch'] = info_dict

    def on_train_end(self, trainer, pl_module):
        if self.unit == 'epoch':
            metrics = self.progress_info[0]['train_epoch'].keys()
            train_info = {
                metric: [e['train_epoch'][metric] for e in self.progress_info]
                for metric in metrics
            }
            if len(trainer.datamodule.val_dataloader()) > 0:
                metrics = self.progress_info[0]['val_epoch'].keys()
            else:
                metrics = {}
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
            metrics = self.progress_info[0]['val_epoch'][0].keys()
            val_info = {
                metric: list(chain.from_iterable([[b[metric] for b in e['val_epoch']] for e in self.progress_info if e['val_epoch']]))
                for metric in metrics
            }
            trainer.progress = {'train': train_info, 'val': val_info}
