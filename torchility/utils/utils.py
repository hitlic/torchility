from typing import Iterable
import numpy as np
import itertools
import torch
from matplotlib import pyplot as plt

def detach_clone(tensors):
    if isinstance(tensors, torch.Tensor):
        return tensors.detach().clone()
    else:
        return [detach_clone(t) for t in tensors]

def concat(tensor_lst, dim=0):
    if isinstance(tensor_lst[0], (list, tuple)):
        return [torch.concat(ts, dim) for ts in zip(*tensor_lst)]
    else:
        return torch.concat(tensor_lst, dim)

class TensorTuple(tuple):
    """
    list of tensors
    """
    @property
    def device(self):
        if len(self) > 0:
            return self[0].device
        else:
            return torch.device(type='cpu')

    def to(self, device, **kwargs):
        return TensorList(t.to(device, **kwargs) for t in self)

    def cpu(self):
        return TensorList(t.cpu() for t in self)

    def clone(self):
        return TensorList(t.clone() for t in self)

    def detach(self):
        return TensorList(t.detach() for t in self)

    @property
    def data(self):
        return TensorList(t.data for t in self)

    def float(self):
        return TensorList(t.float() for t in self)

    def long(self):
        return TensorList(t.long() for t in self)

    def int(self):
        return TensorList(t.int() for t in self)

TensorList = TensorTuple

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator


def listify(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, Iterable):
        return list(obj)
    return [obj]


def plot_confusion(c_matrix, class_num, class_names=None,
                   normalized=False, norm_dec=2, cmap='Blues', info=''):
    """
    画出混淆矩阵。
    Args:
        c_matrix: 混淆矩阵
        class_num: 类别数量
        class_names: 各类名称，可选参数
        normalized: c_matrix是否经过标准化
        norm_dec: 标准化保留小数点位数
        cmap: 配色方案
        info: 显示在图像标题中的其他信息
    """
    title = 'Confusion matrix'

    data_size = c_matrix.sum()
    if not normalized:
        c_matrix = c_matrix.astype('int')

    fig = plt.figure()

    plt.imshow(c_matrix, interpolation='nearest', cmap=cmap)
    plt.title(f'{title} - ({data_size}) \n{info}')
    if class_names and len(class_names) == class_num:
        tick_marks = np.arange(class_num)
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names, rotation=0)

    thresh = c_matrix.max() / 2.
    for i, j in itertools.product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
        coeff = f'{c_matrix[i, j]:.{norm_dec}f}' if normalized else f'{c_matrix[i, j]}'
        plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center",
                 color="white" if c_matrix[i, j] > thresh else "black")

    ax = fig.gca()
    ax.set_ylim(class_num-.5, -.5)

    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.grid(False)
    # plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    return fig

