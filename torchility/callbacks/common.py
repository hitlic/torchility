from pytorch_lightning.callbacks.callback import Callback
import numpy as np
from ..utils import flatten_dict

class ResetMetrics(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        metric_values = {}
        if pl_module.metrics:
            metric_values = reset_metrics('train', pl_module)
        trainer.train_epoch_metric_values = metric_values

    def on_validation_epoch_end(self, trainer, pl_module):
        metric_values = {}
        if pl_module.metrics:
            metric_values = reset_metrics('val', pl_module)
        trainer.val_epoch_metric_values = metric_values

    def on_test_epoch_end(self, trainer, pl_module):
        metric_values = {}
        if pl_module.metrics:
            metric_values = reset_metrics('test', pl_module)
        trainer.test_epoch_metric_values = metric_values


def reset_metrics(stage, task):    # 重置指标
    metrics = task.metrics[stage]
    metric_values = {}
    for metric in metrics:
        value = metric.compute()
        if isinstance(value, dict):
            for k, v in value.items():
                metric_values[dict_metric_name(metric.name, k)] = v
            log_values = {f'{stage}_{k}': v for k, v in metric_values.items()}
            task.log_dict(log_values, prog_bar=metric.on_bar, on_step=False, on_epoch=metric.log)
        else:
            metric_values[metric.name] = value
            task.log(f'{stage}_{metric.name}', value, prog_bar=metric.on_bar, on_step=False, on_epoch=metric.log)
        metric.reset()
    return metric_values


def dict_metric_name(m_name, dict_key):
    if '/' in m_name:
        return m_name.replace('/', f'{dict_key}/')
    else:
        return f'{m_name}{dict_key}'


class LogHyperParameters(Callback):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = flatten_dict(hparams)

    def on_fit_end(self, trainer, pl_module):
        metric_values = self.get_fit_metrics(trainer.progress)
        trainer.logger.log_hyperparams(self.hparams, metric_values)
        return super().on_fit_end(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        metric_values = {f'test_{k}': float(v) for k, v in trainer.test_metrics.items()}
        trainer.logger.log_hyperparams(self.hparams, metric_values)
        return super().on_test_end(trainer, pl_module)

    def get_fit_metrics(self, progress):
        m_values = {}
        if 'val' in progress:   # 如果有验证集，则获取验证损失最小的epoch相应的指标
            idx = np.array(progress['val']['loss']).argmin()
        else:                   # 如果无验证集，则获取训练损失最小的epoch相应的指标
            idx = np.array(progress['train']['loss']).argmin()

        for m, vs in progress['train'].items():
            m_values[f'train_{m}'] = float(vs[idx])
        if 'val' in progress:
            for m, vs in progress['val'].items():
                m_values[f'val_{m}'] = float(vs[idx])
        return m_values
