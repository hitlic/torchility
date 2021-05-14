from pytorch_lightning import LightningModule


class _TorchilityModule(LightningModule):
    def __init__(self, model, loss, optimizer, log_step=True, log_epoch=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.opt = optimizer
        self.on_step = log_step
        self.on_epoch = log_epoch

    def forward(self, batch_data):              # 前向计算
        return self.model(batch_data)

    def training_step(self, batch, batch_nb):   # 训练步
        model_input, label = batch
        model_out = self(model_input)
        loss = self.loss_fn(model_out, label)
        self.log('train_loss', loss, on_step=self.on_step, on_epoch=self.on_epoch)
        return loss

    def validation_step(self, batch, batch_nb):
        model_input, label = batch
        model_out = self(model_input)
        loss = self.loss_fn(model_out, label)
        self.log('val_loss', loss, prog_bar=True, on_step=self.on_step, on_epoch=self.on_epoch)
        return {'val_loss': loss}

    def test_step(self, batch, batch_nb):
        model_input, label = batch
        model_out = self(model_input)
        loss = self.loss_fn(model_out, label)
        self.log('test_loss', loss)
        return loss

    def predict(self, batch, batch_idx, dataloader_idx=None):
        model_input, _ = batch
        model_out = self(model_input)
        return model_out

    def configure_optimizers(self):
        return self.opt
