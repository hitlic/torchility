from pytorch_lightning.callbacks.base import Callback
from ..hooks import Hooks
import matplotlib.pyplot as plt


class LRFinder(Callback):
    def __init__(self, max_batch=100, min_lr=1e-6, max_lr=10):
        super().__init__()
        self.max_batch, self.min_lr, self.max_lr = max_batch, min_lr, max_lr
        self.best_loss = 1e9
        self.lrs = []
        self.losses = []

    def on_fit_start(self, trainer, pl_module):
        trainer.lrfinder = self

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        pos = (batch_idx + 1)/self.max_batch
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in pl_module.opt.param_groups:
            pg['lr'] = lr
        self.lrs.append(lr)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        batch_loss = outputs['loss']
        if batch_idx+1 >= self.max_batch or batch_loss > self.best_loss*10:
            trainer.should_stop = True
        if batch_loss < self.best_loss:
            self.best_loss = batch_loss
        self.losses.append(batch_loss)

    def plot(self):
        plt.plot(self.lrs, self.losses)
        plt.xlabel('learning rate')
        plt.ylabel('loss')


class ModelAnalyzer(Callback):
    def __init__(self, mode='backward'):
        """
        Args:
            mode: 'forward' 分析各层前向输出 or 'backward' 分析各层反向梯度
        """
        super().__init__()
        assert mode in ['forward', 'backward'], '`mode` must be "forward" or "backward"!'
        self.mode = mode

    def on_fit_start(self, trainer, pl_module):
        def output_stats(hook, module, inputs, outputs):
            if isinstance(outputs, tuple):  # backward hook
                outputs = outputs[0]
            hook.mean = outputs[0].data.mean()
            hook.std = outputs[0].data.std()
            hook.data = outputs[0].data
        self.hooks = Hooks(pl_module.model, output_stats, self.mode)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        mode = 'FORWARD' if self.mode == 'forward' else 'BACKWARD'
        mean_dict = {h.name: h.mean for h in self.hooks}
        std_dict = {h.name: h.std for h in self.hooks}
        logger = pl_module.logger.experiment
        logger.add_scalars(f'{mode}-mean', mean_dict, global_step=pl_module.global_step)
        logger.add_scalars(f'{mode}-std', std_dict, global_step=pl_module.global_step)
        for h in self.hooks:
            logger.add_histogram(f'{mode}-{h.name}', h.data, global_step=pl_module.global_step)

    def on_fit_end(self, trainer, pl_module):
        self.hooks.remove()
