from pytorch_lightning import LightningModule
from torchmetrics import Metric
from .utils import detach_clone
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

class GeneralTaskModule(LightningModule):
    def __init__(self, model, loss, optimizer, metrics=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.loss_fn = loss
        self.opt = optimizer
        self.metrics = metrics
        self.messages = dict()          # 存放训练、验证、测试过程中的各种消息数据
        self.do_test_loss = True

    def forward(self, *batch_data):                         # 前向计算
        return self.model(*batch_data)

    def on_train_start(self):
        self.metrics ={
            'train': [m.to(self.device) for m in self.metrics['train'] if isinstance(m, Metric)],
            'val': [m.to(self.device) for m in self.metrics['val'] if isinstance(m, Metric)],
            'test': [m.to(self.device) for m in self.metrics['test'] if isinstance(m, Metric)]
        }

    def on_test_start(self):
        self.metrics['test'] = [m.to(self.device) if isinstance(m, Metric) else m for m in self.metrics['test']]


    def training_step(self, batch, batch_nb):               # 训练步
        loss, preds, targets, batch_size = self.do_forward(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)

        preds, targets = detach_clone(preds), detach_clone(targets)
        if self.metrics:
            self.do_metric(preds, targets, 'train', True, True)
        self.messages['train_batch'] = (batch_nb, preds, targets)  # (batch_idx, preds, tagets)
        return loss

    def validation_step(self, batch, batch_nb):             # 验证步
        loss, preds, targets, batch_size = self.do_forward(batch)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)

        preds, targets = detach_clone(preds), detach_clone(targets)
        if self.metrics:
            self.do_metric(preds, targets, 'val', True, True)
        self.messages['val_batch'] = (batch_nb, preds, targets)  # (batch_idx, preds, tagets)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):                   # 测试步
        loss, preds, targets, batch_size = self.do_forward(batch, self.do_test_loss)
        if loss is not None:
            self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)

        preds, targets = detach_clone(preds), detach_clone(targets)
        if self.metrics:
            self.do_metric(preds, targets, 'test', True, True)
        self.messages['test_batch'] = (batch_nb, preds, targets)  # (batch_idx, preds, tagets)
        if loss is not None:
            return {'test_loss': loss}
        else:
            return None

    def configure_optimizers(self):                         # 优化器
        if isinstance(self.opt, (list, tuple)) and len(self.opt) == 2 \
            and isinstance(self.opt[0], Optimizer) \
            and isinstance(self.opt[1], _LRScheduler):
            return {
                'optimizer': self.opt[0],
                'lr_scheduler': self.opt[1]
            }
        else:
            return self.opt

    def do_forward(self, batch, do_loss=True):                            # 前向计算
        input_feat, targets = batch[:-1], batch[-1]  # batch最后一个元素为标签
        preds = self(*input_feat)
        if do_loss:
            loss = self.loss_fn(preds, targets)
        else:
            loss = None
        batch_size = batch[0].shape[0] if isinstance(batch, (tuple, list)) else batch.shape[0]
        return loss, preds, targets, batch_size

    def do_metric(self, preds, targets, state, on_step, on_epoch):  # 指标计算
        metrics = self.metrics[state]
        for metric in metrics:
            if isinstance(metric, Metric):
                metric(preds, targets)
                self.log(f"{state}_{metric.name}", metric, prog_bar=True, on_step=on_step, on_epoch=on_epoch, metric_attribute=metric, batch_size=preds.shape[0])
            else:
                value = metric(preds, targets)
                self.log(f"{state}_{metric.name}_step", value, prog_bar=True, on_step=on_step, on_epoch=False, batch_size=preds.shape[0])
