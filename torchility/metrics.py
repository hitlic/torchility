import torch
from .utils import rename, concat, get_batch_size
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class MetricNotUpdated(Exception):
    pass


class MetricBase:
    """
    torchility评价指标的使用方式有如下三种：
    （1）torchmetrics.Metric的子类，包括torchmetrics中提供的各种指标。
        如果模型输出结果比较复杂，可重写forward方法，先预处理再调用父类forward方法。
    （2）自定义指标。继承MetricBase，指标的计算方法可以以metric_fn参数传入，或者重写forward方法。
        如果模型输出结果比较复杂，可重写prepare方法预处理。
    （3）提供一个函数等可调用对象作为评价指标，它会被自动封装为MetricBase的对象，运行过程与（2）相同。
    
        这三种方式中，最推荐使用第（1）种方式，因为后两种方式会把每个batch的预测和标签记录下来以计算epoch的结果，
    可能占用较大存储空间。

    注意：__建议__在torchility.Trainer的metrics参数中，为每个指标指定一个名字，用于在进度中显示。例如，
        m1 = ('m1_name', metric1)            或者 {'name': 'm1_name', 'metric': metric1}
        m2 = 'm2_name', metric2              或者 {'name': 'm2_name', 'metric': metric2}
        trainer = torchility.Trainer(metrics=[m1, m2], ...)
    """
    def __init__(self, metric_fn=None, name=None, simple_cumulate=False):
        """
        Args:
            metric_fn: 计算指标的函数
            name: 指标名称，用于进度显示
            simple_cumulate: True：简单地利用batch_size自动累积计算epoch上的指标，
                                   但利用混淆矩阵计算的指标这种方式会导致epoch指标计算错误；
                             False：保存每个batch的预测和标签，在每个epoch之后计算。
        """
        self.name = self.__class__.__name__ if name is None else name
        if metric_fn:
            self.metric_fn = metric_fn
        else:
            self.metric_fn = self.forward
        self.simple_cumulate = simple_cumulate

        self.pred_batchs = []
        self.target_batchs = []

        self.cumulate_size = 0
        self.cumulate_value = None

    def __call__(self, preds, targets):
        return self.update(preds, targets)

    def prepare(self, preds, targets):
        """如果模型输出结果或者标签比较复杂，在计算指标前需要预处理，则需要重写本方法"""
        return preds, targets

    def forward(self, preds, targets):
        return NotImplemented

    def update(self, preds, targets):
        preds, targets = self.prepare(preds, targets)
        m_value = self.metric_fn(preds, targets)

        if self.simple_cumulate:
            if self.cumulate_value is None:
                self.cumulate_value = {k: 0.0 for k in m_value.keys()} if isinstance(m_value, dict) else 0.0
            b_size = get_batch_size(preds)
            self.cumulate_size += b_size
            if isinstance(m_value, dict):
                for k, v in m_value.items():
                    self.cumulate_value[k] += b_size * v
            else:
                self.cumulate_value += b_size * m_value
        else:
            self.pred_batchs.append(preds)
            self.target_batchs.append(targets)
        return m_value

    def reset(self):
        self.pred_batchs = []
        self.target_batchs = []
        self.cumulate_size = 0
        self.cumulate_value = None

    def compute(self):
        if self.simple_cumulate:
            if self.cumulate_size == 0 or self.cumulate_value is None:
                raise MetricNotUpdated("The metric have not been updated!")
            if isinstance(self.cumulate_value, dict):
                return {k: v/self.cumulate_size for k, v in self.cumulate_value.items()}
            else:
                return self.cumulate_value / self.cumulate_size
        else:
            if len(self.pred_batchs) == 0 and len(self.pred_batchs) == 0:
                return 0.0000
            pred_epoch = concat(self.pred_batchs)
            target_epoch = concat(self.target_batchs)
            return self.metric_fn(pred_epoch, target_epoch)

    def clone(self):
        return deepcopy(self)


@rename('acc')
def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()


@rename('acc')
def masked_accuracy(preds, targets):
    _, preds = preds.max(dim=1)
    correct = preds[targets.mask].eq(targets.data[targets.mask]).sum()
    acc = correct / targets.mask.float().sum()
    return acc


@rename('mse')
def masked_mse(preds, targets):
    return F.mse_loss(torch.squeeze(preds[targets.mask]), targets.data[targets.mask])


@rename('mae')
def masked_mae(preds, targets):
    preds = torch.squeeze(preds[targets.mask])
    targets = targets.data[targets.mask]
    return torch.mean(torch.abs(preds - targets))


def ordinal(preds, targets):
    """
    真实排序下，对应的预测值的序号
    """
    frame = np.array([preds, targets]).transpose()
    frame = frame[(-frame[:, 0]).argsort()]
    frame = np.concatenate([frame, np.expand_dims(np.arange(1, len(preds)+1), 1)], 1)
    frame = frame[(-frame[:, 1]).argsort()]
    preds_ids = frame[:, 2]
    return preds_ids


@rename('map')
def MAP(preds, targets):
    """
    Mean average precision(MAP)
    """
    n = len(preds)
    targets_ids = np.arange(1, len(preds)+1)
    preds_ids = ordinal(preds, targets)

    def p_at_n(p_ids, t_ids, n):
        return len(set(p_ids[:n]).intersection(set(t_ids[:n])))/n
    return np.average([p_at_n(preds_ids, targets_ids, i) for i in range(1, n+1)])


@rename('map')
def masked_MAP(preds, targets):
    """
    Mean average precision(MAP)
    """
    preds = torch.squeeze(preds[targets.mask]).detach().cpu().numpy()
    targets = targets.data[targets.mask].detach().cpu().numpy()
    return MAP(preds, targets)


def DCG_at_n(preds, targets, n):
    """
    Discount Cumulative Gain (DCG@n)
    """
    frame = np.array([preds, targets]).transpose()
    frame = frame[(-frame[:, 0]).argsort()]
    frame = frame[:n]
    return np.sum([t/np.log2(i+2) for i, (_, t) in enumerate(frame)])


def NDCG_at_n(n, preds, targets):
    """
    Normalized discount cumulative gain (NDCG@n)
    """
    return DCG_at_n(preds, targets, n)/DCG_at_n(targets, targets, n)


def masked_NDCG_at_n(n, preds, targets):
    """
    Masked normalized discount cumulative gain (NDCG@n)
    """
    preds = torch.squeeze(preds[targets.mask]).detach().cpu().numpy()
    targets = targets.data[targets.mask].detach().cpu().numpy()
    return NDCG_at_n(n, preds, targets)
