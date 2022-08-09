
import matplotlib.pyplot as plt
import math
from collections import deque
from itertools import chain


class SeriesPlots:
    def __init__(self, plots, colums=1, figsize=None):
        """
        画动态时间序列数据的工具。
        Args:
            plots: 用于指定包含多少个子图，各子图中有几条序列，以及各序列的名字。
                    例如 plots = ['s_name1', ['s_name21', 's_name22']]，表示包含两个子图。
                    第一个子图中有一条名为s_name1的序列，第二个子图中有两条名分别为s_name21和s_name22的序列。
            colums: 子图列数
            fig_size: 图像大小
        Example:
            import random
            import time
            sp = SeriesPlots([ 'x', ['xrand', 'xrand_move']], 2)
            mv = Moveing(10)
            for i in range(1000):
                sp.add(i, [i+random.random()*0.5*i, mv(i+random.random()*0.5*i)])
                time.sleep(0.1)
        """
        self.plot_names = plots
        self.fig_size = figsize
        self.colums = min(len(plots), colums)
        self.rows = 1
        if len(plots) > 1:
            self.rows = math.ceil(len(plots)/colums)

        self.x = []
        self.ys = []
        self.graphs = []

        self.is_start = False
        self.max_ys = [-1000000000] * len(plots)
        self.min_ys = [1000000000] * len(plots)

    def ioff(self):
        """关闭matplotlib交互模式"""
        plt.ioff()

    def start(self):
        plt.ion()
        self.fig, axs = plt.subplots(self.rows, self.colums, figsize=self.fig_size)
        if len(self.plot_names) == 1:
            axs = [axs]
        if self.rows > 1 and self.colums > 1:
            axs = chain(*axs)
        for s_name, ax in zip(self.plot_names, axs):
            if isinstance(s_name, (list, tuple)):
                gs = []
                yy = []
                for n in s_name:
                    ln, = ax.plot([], [], label=n)
                    gs.append((ax, ln))
                    yy.append([])
                self.graphs.append(gs)
                self.ys.append(yy)
            else:
                ln, = ax.plot([], [], label=s_name)
                self.graphs.append((ax, ln))
                self.ys.append([])
            ax.legend()

    def check_shape(self, lstx, lsty):
        spx = [len(item) if isinstance(item, (list, tuple)) else 0 for item in lstx]
        spy = [len(item) if isinstance(item, (list, tuple)) else 0 for item in lsty]
        return spx == spy

    def add(self, *values):
        """
        values的形状必须和 __init__ 中的plots参数形状匹配。
        """
        if not self.is_start:
            self.start()
            self.is_start = True
        assert self.check_shape(values, self.plot_names), f'数据形状不匹配！应当形如：{str(self.plot_names)[1:-1]}'
        self.x.append(len(self.x) + 1)

        for i, (ys, vs, axs_lns) in enumerate(zip(self.ys, values, self.graphs)):
            if isinstance(vs, (list, tuple)):
                self.max_ys[i] = max(self.max_ys[i], max(vs))
                self.min_ys[i] = min(self.min_ys[i], min(vs))
                for y, v, (ax, ln) in zip(ys, vs, axs_lns):
                    y.append(v)
                    ln.set_xdata(self.x)
                    ln.set_ydata(y)
            else:
                self.max_ys[i] = max(self.max_ys[i], vs)
                self.min_ys[i] = min(self.min_ys[i], vs)
                ax, ln = axs_lns
                ys.append(vs)
                ln.set_xdata(self.x)
                ln.set_ydata(ys)
            ax.set_xlim(0, len(self.x) + 1)
            ax.set_ylim(min(0, self.min_ys[i]) - 0.05 * math.fabs(self.min_ys[i]) - 0.1,
                        self.max_ys[i]+0.05*math.fabs(self.max_ys[i]))
            # plt.pause(0.05)
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.05)


def Moveing(length=10):
    """
    返回一个能够计算滑动平均的函数。
    length: 滑动平均长度
    """
    values = deque(maxlen=length)

    def moveing_average(v):
        values.append(v)
        return sum(values)/len(values)
    return moveing_average
