from pytorch_lightning.callbacks.base import Callback
from ..metrics import MetricBase

class ResetMetrics(Callback):
    def __init__(self):
        super().__init__()
    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.metrics:
            self.reset_metrics('train', pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.metrics:
            self.reset_metrics('val', pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if pl_module.metrics:
            self.reset_metrics('test', pl_module)

    def reset_metrics(self, state, task):                         # 重置指标
        metrics = task.metrics[state]
        for metric in metrics:
            if not isinstance(metric, MetricBase): continue
            value = metric.compute()
            task.log(f"{state}_{metric.name}_epoch", value, prog_bar=True, on_step=False, on_epoch=True)
            metric.reset()
