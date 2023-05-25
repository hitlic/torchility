from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.classification import BinaryConfusionMatrix, MulticlassConfusionMatrix
from itertools import chain
import torch
from ..utils import plot_confusion, TopKQueue


class Interpreter(Callback):
    def __init__(self, metric=None, k=100, mode='max', stage='test'):
        """
        Args:
            metric: none reducted callable
            k: number of samples to keep
            mode: 'max' or 'min'
            stage: 'train', 'val' or 'test'
        """
        super().__init__()
        assert mode in ['min', 'max']
        assert stage in ['train', 'val', 'test']
        self.metric = metric
        self.stage = stage
        self.mode = mode
        self.batch_recorder = []
        self.top_queue = TopKQueue(k=k)

    def on_fit_start(self, trainer, pl_module):
        trainer.interpreter = self
        pl_module.interpreter = self

    def on_train_epoch_start(self, trainer, pl_module):
        self.batch_recorder = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.batch_recorder = []

    def on_test_epoch_start(self, trainer, pl_module):
        self.batch_recorder = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.stage == 'train':
            self.batch_recorder.append(pl_module.messages['train_batch'])
            _, preds, targets = pl_module.messages['train_batch']
            self.put_queue(preds, targets, *batch[:-1])

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.stage == 'val':
            self.batch_recorder.append(pl_module.messages['val_batch'])
            _, preds, targets = pl_module.messages['val_batch']
            self.put_queue(preds, targets, *batch[:-1])

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.stage == 'test':
            self.batch_recorder.append(pl_module.messages['test_batch'])
            _, preds, targets = pl_module.messages['test_batch']
            self.put_queue(preds, targets, *batch[:-1])

    def put_queue(self, preds, targets, *inputs):
        if self.metric is None:
            return
        batch_m = self.metric(preds, targets)
        assert batch_m.shape[0] == preds.shape[0], 'The `metric` must not be reduced!'
        for m, pred, target, *xs in zip(batch_m.cpu(), preds.cpu(), targets.cpu(), *[ipt.cpu() for ipt in inputs]):
            v = -m if self.mode == 'min' else m
            feat = xs[0].detach().numpy() if len(xs)==1 else [x.detach().numpy() for x in xs]
            self.top_queue.put((v.item(), [m.item(), pred.detach().numpy(), target.detach().numpy(), feat]))

    def top_samples(self):
        """
        Return: [metric_value, pred, target, input_feat] of top k samples 
        """
        samples = [item[1] for item in self.top_queue.items()]
        return samples

    def top(self):
        return self.top_samples()


class ClassifierInterpreter(Interpreter):
    """
    分类模型的解释器
    """
    def __init__(self, class_num, binary=False, normalize=False, metric=None, k=0, mode='max', stage='test', threshold=0.5):
        """
        class_num: number of class.
        binary: True of False
        normalize: normalize or not
        metric: none reducted callable
        k: number of samples
        mode: 'max' or 'min'
        stage: 'train', 'val' or 'test'
        threshold: threshold of classification
        """
        super().__init__(metric, k=k, mode=mode, stage=stage)
        if binary:
            assert class_num == 2, "class_mnum must be 2 for binary classification!"
        norm = 'true' if normalize else 'none'
        if binary:
            self.conf_mat = BinaryConfusionMatrix(normalize=norm, threshold=threshold)
        else:
            self.conf_mat = MulticlassConfusionMatrix(class_num, normalize=norm, threshold=threshold)
        self.class_num = class_num
        self.normalize = normalize

    def confusion_matrix(self, argmax_pred=False, argmax_target=False):
        for _, preds, targets in self.batch_recorder:
            b_preds = preds.argmax(dim=1) if argmax_pred else preds
            b_targets = targets.argmax(dim=1) if argmax_target else targets
            self.conf_mat.update(b_preds.cpu(), b_targets.cpu())
        return self.conf_mat.compute().detach().numpy()

    def plot_confusion(self, class_names=None, cmap='Blues', title_info='', argmax_pred=False, argmax_target=False):
        c_matrix = self.confusion_matrix(argmax_pred, argmax_target)
        return plot_confusion(c_matrix, self.class_num, class_names, self.normalize, cmap=cmap, info=title_info)
