from pytorch_lightning import LightningModule
from torchmetrics import Metric

class GeneralTaskModule(LightningModule):
    def __init__(self, model, loss, optimizer, metrics=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.loss_fn = loss
        self.opt = optimizer
        self.metrics = metrics
        self.messages = dict()          # 存放训练、验证、测试过程中的各种消息数据

    def forward(self, *batch_data):                         # 前向计算
        return self.model(*batch_data)

    def training_step(self, batch, batch_nb):               # 训练步
        loss, preds, targets = self.do_forward(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        if self.metrics:
            self.do_metric(preds, targets, 'train', True, True)
        self.messages['train_batch'] = (batch_nb, preds, targets)  # (batch_idx, preds, tagets)
        return loss

    def validation_step(self, batch, batch_nb):             # 验证步
        loss, preds, targets = self.do_forward(batch)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        if self.metrics:
            self.do_metric(preds, targets, 'val', True, True)
        self.messages['val_batch'] = (batch_nb, preds, targets)  # (batch_idx, preds, tagets)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):                   # 测试步
        loss, preds, targets = self.do_forward(batch)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        if self.metrics:
            self.do_metric(preds, targets, 'test', False, True)
        self.messages['test_batch'] = (batch_nb, preds, targets)  # (batch_idx, preds, tagets)
        return {'test_loss': loss}

    def configure_optimizers(self):                         # 优化器
        if isinstance(self.opt, (list, tuple)):
            assert len(self.opt) == 2, '"optimizer" must be an optimizer or a tuple (optimizer, scheduler)!'
            return {
                'optimizer': self.opt[0],
                'lr_scheduler': self.opt[1]
            }
        else:
            return self.opt

    def do_forward(self, batch):                            # 前向计算
        input_feat, targets = batch[:-1], batch[-1]  # batch最后一个元素为标签
        preds = self(*input_feat)
        loss = self.loss_fn(preds, targets)
        return loss, preds, targets

    def do_metric(self, preds, targets, state, on_step, on_epoch):               # 指标计算
        metrics = self.metrics[state]
        for metric in metrics:
            if isinstance(metric, Metric):
                metric(preds, targets.int())
                self.log(f"{state}_{metric.name}", metric, prog_bar=True, on_step=on_step, on_epoch=on_epoch, metric_attribute=metric)
            else:
                value = metric(preds, targets)
                self.log(f"{state}_{metric.name}", value, prog_bar=True, on_step=on_step, on_epoch=on_epoch)
