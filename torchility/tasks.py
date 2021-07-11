from pytorch_lightning import LightningModule
from .metrics import MetricBase


class GeneralTaskModule(LightningModule):
    def __init__(self, model, loss, optimizer, log_step_loss=True, log_epoch_loss=None, metrics=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.opt = optimizer
        self.on_step = log_step_loss
        self.on_epoch = log_epoch_loss
        self.metrics = [] if metrics is None else metrics

        # 存放训练、验证、测试过程中的各种消息数据
        self.messages = dict()

    def forward(self, *batch_data):                         # 前向计算
        return self.model(*batch_data)

    def training_step(self, batch, batch_nb):               # 训练步
        loss, preds, targets = self.do_forward(batch)
        self.log('train_loss', loss, on_step=self.on_step, on_epoch=self.on_epoch)
        self.do_metric(preds, targets, 'train', self.log)
        self.messages['train_batch'] = (batch_nb, preds, targets) # (batch_idx, preds, tagets)
        return loss

    def validation_step(self, batch, batch_nb):             # 验证步
        loss, preds, targets = self.do_forward(batch)
        self.log('val_loss', loss, prog_bar=True, on_step=self.on_step, on_epoch=self.on_epoch)
        self.do_metric(preds, targets, 'val', self.log)
        self.messages['val_batch'] = (batch_nb, preds, targets) # (batch_idx, preds, tagets)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):                   # 测试步
        loss, preds, targets = self.do_forward(batch)
        self.log('test_loss', loss, prog_bar=True)
        self.do_metric(preds, targets, 'test', self.log)
        self.messages['test_batch'] = (batch_nb, preds, targets) # (batch_idx, preds, tagets)
        return {'test_loss': loss}

    def configure_optimizers(self):                         # 优化器
        return self.opt

    def do_forward(self, batch):                            # 前向计算
        input_feat, targets = batch[:-1], batch[-1]  # batch最后一个元素为标签
        preds = self(*input_feat)
        loss = self.loss_fn(preds, targets)
        return loss, preds, targets

    def do_metric(self, preds, targets, state, log):               # 指标计算
        for metric in self.metrics:
            result = metric(preds, targets)
            if isinstance(metric, MetricBase):
                name = metric.name
                on_step = metric.log_step
                on_epoch = metric.log_epoch
            else:
                name = metric.__name__
                on_step, on_epoch = True, True
            log(f"{state}_{name}", result, prog_bar=True, on_step=on_step, on_epoch=on_epoch)
