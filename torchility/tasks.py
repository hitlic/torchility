from pytorch_lightning import LightningModule
from torchmetrics import Metric
from .utils import detach_clone, get_batch_size
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer, Adam
from typing import Any
from .callbacks.common import dict_metric_name


def dfault_loss(preds, targets):
    """
    default loss function, just return the model prediction.
    """
    return preds

class GeneralTaskModule(LightningModule):
    def __init__(self, model, loss=None, optimizer=None, metrics=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        if loss is not None:
            self.loss_fn = loss
        else:
            self.loss_fn = dfault_loss
            print("\033[0;34;m**WARNING: The default loss function is used. Make sure the model returns a loss value.\033[0m")
        if optimizer is not None:
            self.opt = optimizer
        else:
            self.opt = Adam(model.parameters(), lr=0.001)
            print("\033[0;34;m**WARNING: The Adam optimizer is used with learning rate of 0.001.\033[0m")
        self.metrics = metrics
        self.messages = dict()          # 存放训练、验证、测试过程中的各种消息数据
        self.pred_dataloader_has_label = True
        self.current_dl_idx = 0

    def forward(self, *batch_data):                         # 前向计算
        return self.model(*batch_data)

    def on_train_start(self):
        self.metrics ={
            'train': [m.to(self.device) if isinstance(m, Metric) else m for m in self.metrics['train']],
            'val': [m.to(self.device) if isinstance(m, Metric) else m for m in self.metrics['val']],
            'test': [m.to(self.device) if isinstance(m, Metric) else m for m in self.metrics['test']]
        }

    def on_test_start(self):
        self.metrics['test'] = [m.to(self.device) if isinstance(m, Metric) else m for m in self.metrics['test']]


    def training_step(self, batch, batch_nb):               # 训练步
        loss, preds, targets, batch_size = self.do_forward(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)

        preds, targets = detach_clone(preds), detach_clone(targets)
        if self.metrics:
            metric_values = self.do_metric(preds, targets, 'train', 0)
        self.messages['train_batch'] = (batch_nb, preds, targets)  # (batch_idx, preds, tagets)
        metric_values['loss'] = loss
        return metric_values

    def validation_step(self, batch, batch_nb, dataloader_idx=0):             # 验证步
        do_loss = self.do_val_loss
        if isinstance(do_loss, (list, tuple)):
            do_loss = dataloader_idx in do_loss
        loss, preds, targets, batch_size = self.do_forward(batch, do_loss)

        metric_values = {}
        if loss is not None:
            self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
            if self.multi_val_dataloaders:
                metric_values[f'loss/{dataloader_idx}'] = loss
            else:
                metric_values['loss'] = loss

        preds, targets = detach_clone(preds), detach_clone(targets)
        if self.metrics:
            dl_idx = dataloader_idx if self.multi_val_dataloaders else -1
            metric_values.update(self.do_metric(preds, targets, 'val', dl_idx))
        self.messages['val_batch'] = (batch_nb, preds, targets)  # (batch_idx, preds, tagets)

        return metric_values

    def test_step(self, batch, batch_nb, dataloader_idx=0):                   # 测试步
        do_loss = self.do_test_loss
        if isinstance(do_loss, (list, tuple)):
            do_loss = dataloader_idx in do_loss
        loss, preds, targets, batch_size = self.do_forward(batch, do_loss)

        metric_values = {}
        if loss is not None:
            self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
            if self.multi_test_dataloaders:
                metric_values[f'loss/{dataloader_idx}'] = loss
            else:
                metric_values['loss'] = loss

        preds, targets = detach_clone(preds), detach_clone(targets)
        if self.metrics:
            dl_idx = dataloader_idx if self.multi_test_dataloaders else -1
            metric_values.update(self.do_metric(preds, targets, 'test', dl_idx))
        self.messages['test_batch'] = (batch_nb, preds, targets)  # (batch_idx, preds, tagets)

        return metric_values

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        _, preds, _, _ = self.do_forward(batch, False, self.pred_dataloader_has_label)
        return preds

    def configure_optimizers(self):                         # 优化器
        if isinstance(self.opt, (list, tuple)) and len(self.opt) == 2 \
            and isinstance(self.opt[0], Optimizer) \
            and isinstance(self.opt[1], LRScheduler):
            return {
                'optimizer': self.opt[0],
                'lr_scheduler': self.opt[1]
            }
        else:
            return self.opt

    def do_forward(self, batch, do_loss=True, has_label=True):                            # 前向计算
        if has_label:
            input_feat, targets = batch[:-1], batch[-1]  # batch最后一个元素为标签
        else:
            input_feat, targets = batch, None
            if not isinstance(input_feat, (list, tuple)):
                input_feat = [input_feat]
        preds = self(*input_feat)
        if do_loss:
            loss = self.loss_fn(preds, targets)
        else:
            loss = None
        batch_size = get_batch_size(input_feat)
        return loss, preds, targets, batch_size

    def do_metric(self, preds, targets, stage, dl_idx):  # 指标计算
        metrics = self.metrics[stage]
        metric_values = {}
        for metric in metrics:
            if (metric.dl_idx >=0 and metric.dl_idx != dl_idx) or (stage not in metric.stages):
                continue
            value = metric(preds, targets)
            if isinstance(value, dict):  # 一个函数中计算多个指标，返回一个字典
                for k, v in value.items():
                    check_metric_value(v)
                    metric_values[dict_metric_name(metric.name, k)] = v
            else:
                check_metric_value(value)
                metric_values[f"{metric.name}"] = value
        return metric_values


def check_metric_value(value):
    if isinstance(value, (float, int)) or value.numel()==1:
        return
    raise MetricValueError("metric value must be a sigle number!")


class MetricValueError(Exception):
    pass
