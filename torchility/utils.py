import os
import os.path as osp
from typing import Iterable
import yaml
import numpy as np
import itertools
import torch
from matplotlib import pyplot as plt


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


def load_yaml(yaml_path):
    """加载yaml配置文件"""
    def _join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.sep.join(seq)

    def _concat(loader, node):
        seq = loader.construct_sequence(node)
        seq = [str(tmp) for tmp in seq]
        return ''.join(seq)

    yaml.add_constructor('!join', _join)
    yaml.add_constructor('!concat', _concat)

    with open(yaml_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return ddict(cfg)


def check_path(path, create=True):
    """检查路径是否存在"""
    if not osp.exists(path):
        if create:
            print(f'Create path "{path}"!')
            os.mkdir(path)
        else:
            raise Exception(f'Path "{path}" does not exists!')


def check_paths(*paths, create=True):
    """检查多个路径是否存在"""
    for path in paths:
        check_path(path, create)


class ddict(dict):
    """
    可以通过“.”访问的字典。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        self[k] = ddict(v)
                    else:
                        self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    self[k] = ddict(v)
                else:
                    self[k] = v

    def __getattr__(self, key):
        try:
            value = self[key]
            return value
        except KeyError:
            raise Exception(f'KeyError! The key "{key}" does not exists!')

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]
