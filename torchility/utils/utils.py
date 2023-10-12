from typing import Iterable
import numpy as np
import itertools
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from queue import PriorityQueue


class TopKQueue(PriorityQueue):
    """
    能够保存最大值的优先队列
    """
    def __init__(self, k: int = 0):
        super().__init__(maxsize=k)

    def put(self, e):
        if self.full():
            if e[0] > self.queue[0][0]:
                self.get()
            else:
                return
        super().put(e)

    def items(self):
        return sorted(self.queue, key=lambda e: e[0], reverse=True)


def batches(inputs, batch_size):
    """
    把inputs按batch_size进行划分
    """
    is_list_input = isinstance(inputs, (list, tuple))  # inputs是否是多个输入组成的列表或元素
    start_idx = 0
    is_over = False
    while True:
        if is_list_input:
            batch = TensorTuple([data[start_idx: start_idx + batch_size] for data in inputs])
            is_over = len(batch[0]) > 0
            start_idx += len(batch[0])
        else:
            batch = inputs[start_idx: start_idx + batch_size]
            is_over = len(batch) > 0
            start_idx += len(batch)
        if is_over > 0:
            yield batch
        else:
            break


def detach_clone(tensors):
    if isinstance(tensors, torch.Tensor):
        return tensors.detach().clone()
    elif isinstance(tensors, (list, tuple)):
        return [detach_clone(t) for t in tensors]
    else:
        return tensors


def concat(tensor_lst, dim=0):
    if isinstance(tensor_lst[0], (list, tuple)):
        return [torch.concat(ts, dim) for ts in zip(*tensor_lst)]
    else:
        return torch.concat(tensor_lst, dim)


def get_batch_size(inputs):
    """检测batch size"""
    if isinstance(inputs, (tuple, list)):
        data = inputs[0]
    else:
        data = inputs
    if hasattr(data, 'shape'):
        return data.shape[0]
    elif hasattr(data, '__len__'):
        return len(data)
    else:
        data_type = f'{type(data).__module__}.{type(data).__name__}'
        if data_type == 'dgl.heterograph.DGLHeteroGraph':
            return data.batch_size
        elif data_type == 'torch_geometric.data.batch.DataBatch':
            return data.num_graphs
        elif data_type == 'torch_geometric.data.data.Data':
            return 1
        else:
            raise ValueError('Unknown batch size!')


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
        return TensorTuple(t.to(device, **kwargs) for t in self)

    def cpu(self):
        return TensorTuple(t.cpu() for t in self)

    def clone(self):
        return TensorTuple(t.clone() for t in self)

    def detach(self):
        return TensorTuple(t.detach() for t in self)

    @property
    def data(self):
        return TensorTuple(t.data for t in self)

    def float(self):
        return TensorTuple(t.float() for t in self)

    def long(self):
        return TensorTuple(t.long() for t in self)

    def int(self):
        return TensorTuple(t.int() for t in self)

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
                 color="yellow" if c_matrix[i, j] > thresh else "green")

    ax = fig.gca()
    ax.set_ylim(class_num-.5, -.5)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.grid(False)
    # plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    return fig


def unpack(data, num=None):
    """
    unpack data to given number of variables.
    Example:
        Code:
            data = [1, 2]
            x, y, z = unpack(data, 3)
            print(x, y, z)
        Output:
            1 2 None

    Args:
        data: data to be unpacked
        num: the number of variables
    """
    if num is None:
        return data
    if len(data) < num:
        data = [d for d in data]
        data.extend([None] * (num - len(data)))
    else:
        data = data[:num]
    return data


def groupby_apply(values: torch.Tensor, keys: torch.Tensor, reduction: str = "mean"):
    """
    Groupby apply for torch tensors.
    Example: 
        Code:
            x = torch.FloatTensor([[1,1], [2,2],[3,3],[4,4],[5,5]])
            g = torch.LongTensor([0,0,1,1,1])
            print(groupby_apply(x, g, 'mean'))
        Output:
            tensor([[1.5000, 1.5000],
                    [4.0000, 4.0000]])
    Args:
        values: values to aggregate - same size as keys
        keys: tensor of groups. 
        reduction: either "mean" or "sum"
    Returns:
        tensor with aggregated values
    """
    if reduction == "mean":
        reduce = torch.mean
    elif reduction == "sum":
        reduce = torch.sum
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")
    keys = keys.to(values.device)
    _, counts = keys.unique(return_counts=True)
    reduced = torch.stack([reduce(item, dim=0) for item in torch.split_with_sizes(values, tuple(counts))])
    return reduced


def flatten_dict(d, parent_key='', sep='.'):
    """flatten a dict with dict as values"""
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
