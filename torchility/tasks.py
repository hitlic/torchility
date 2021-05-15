from pytorch_lightning import LightningModule
from .metrics import MetricBase

class GeneralTask(LightningModule):
    def __init__(self, model, loss, optimizer, log_loss_step=True, log_loss_epoch=None, metrics=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.opt = optimizer
        self.on_step = log_loss_step
        self.on_epoch = log_loss_epoch
        self.metrics = [] if metrics is None else metrics

        # 最新训练、验证、测试batch中的batch_id, preds和targets
        self.train_batch_preds_targets = (0, None, None)
        self.val_batch_preds_targets = (0, None, None)
        self.test_batch_preds_targets = (0, None, None)

    def forward(self, batch_data):                  # 前向计算
        return self.model(batch_data)

    def training_step(self, batch, batch_nb):       # 训练步
        model_input, targets = batch
        preds = self(model_input)
        loss = self.loss_fn(preds, targets)
        self.log('train_loss', loss, on_step=self.on_step, on_epoch=self.on_epoch)
        for metric in self.metrics:
            self.do_metric(metric, preds, targets, self.log)
        self.train_batch_preds_targets = (batch_nb, preds, targets)
        return loss

    def validation_step(self, batch, batch_nb):     # 验证步
        model_input, targets = batch
        preds = self(model_input)
        loss = self.loss_fn(preds, targets)
        self.log('val_loss', loss, prog_bar=True, on_step=self.on_step, on_epoch=self.on_epoch)
        for metric in self.metrics:
            self.do_metric(metric, preds, targets, self.log)
        self.val_batch_preds_targets = (batch_nb, preds, targets)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):           # 测试步
        model_input, targets = batch
        preds = self(model_input)
        loss = self.loss_fn(preds, targets)
        self.log('test_loss', loss, prog_bar=True)
        for metric in self.metrics:
            self.do_metric(metric, preds, targets, self.log)
        self.test_batch_preds_targets = (batch_nb, preds, targets)
        return {'test_loss': loss}

    def configure_optimizers(self):                 # 优化器
        return self.opt

    def do_metric(self, metric, preds, targets, log):
        result = metric(preds, targets)
        if isinstance(metric, MetricBase):
            name = metric.name
            on_step = metric.log_step
            on_epoch = metric.log_epoch
        else:
            name = metric.__name__
            on_step, on_epoch = True, True
        log(name, result, prog_bar=True, on_step=on_step, on_epoch=on_epoch)
