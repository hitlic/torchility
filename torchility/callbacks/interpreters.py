from pytorch_lightning.callbacks.base import Callback
from torchmetrics import ConfusionMatrix
from itertools import chain
import torch
from ..utils import plot_confusion


class BatchRecorder(Callback):
    def __init__(self, stage='test'):
        super().__init__()
        assert stage in ['train', 'val', 'test']
        self.stage = stage

    def on_epoch_start(self, trainer, pl_module):
        self.recorder = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.stage == 'train':
            self.recorder.append(pl_module.messages['train_batch'])

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.stage == 'val':
            self.recorder.append(pl_module.messages['val_batch'])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.stage == 'test':
            self.recorder.append(pl_module.messages['test_batch'])


class Interpreter(BatchRecorder):
    def on_fit_start(self, trainer, pl_module):
        trainer.interpreter = self
        pl_module.interpreter = self

    def top_samples(self, metric, k=0, largest=True):
        """
        返回metric指标最大（largest=True）或最小（largest=False）的k个样本数据的信息。
        Args:
            metric: metric对于每个样本应当只输出一个数值。
            larges: 排序方式
            k: 返回数量
        Return: [(batch_id, sample_id_within_batch), metric_value, pred, target), ... ]
        """
        # 检验 metric 是否正确
        _, preds, targets = self.recorder[0]
        values = metric(preds, targets)
        assert values.size()[0] == preds.size()[0], 'The output of `metric` must not be reduced!'

        info_batchs = []
        with torch.no_grad():
            for batch_id, preds, targets in self.recorder:
                values = metric(preds, targets).flatten().detach().cpu()
                ids = [(batch_id, i) for i in range(values.size()[0])]
                info_batchs.append((ids, values, preds, targets))
        sample_ids = chain(*[e[0] for e in info_batchs])
        sample_values = torch.cat([e[1] for e in info_batchs]).numpy()
        sample_preds = torch.cat([e[2] for e in info_batchs]).numpy()
        sample_targets = torch.cat([e[3] for e in info_batchs]).numpy()
        sample_infos = zip(sample_ids, sample_values, sample_preds, sample_targets)
        sorted_samples = sorted(sample_infos, key=lambda e: e[1], reverse=largest)
        if k > 0:
            sorted_samples = sorted_samples[:k]
        return sorted_samples


class ClassifierInterpreter(Interpreter):
    """
    分类模型的解释器
    """

    def __init__(self, class_num, normalize=False, stage='test'):
        super().__init__(stage=stage)
        norm = 'true' if normalize else 'none'
        self.conf_mat = ConfusionMatrix(class_num, normalize=norm)
        self.class_num = class_num
        self.normalize = normalize

    def confusion_matrix(self, argmax_pred=True, argmax_target=False):
        for _, preds, targets in self.recorder:
            b_preds = preds.argmax(dim=1) if argmax_pred else preds
            b_targets = targets.argmax(dim=1) if argmax_target else targets
            self.conf_mat.update(b_preds.cpu(), b_targets.cpu())
        return self.conf_mat.compute().detach().numpy()

    def plot_confusion(self, class_names=None, cmap='Blues', title_info='', argmax_pred=True, argmax_target=False):
        c_matrix = self.confusion_matrix(argmax_pred, argmax_target)
        return plot_confusion(c_matrix, self.class_num, class_names, self.normalize, cmap=cmap, info=title_info)


class RegressorInterpreter(Interpreter):
    """
    回归模型解释器
    """
    # TODO
    pass
