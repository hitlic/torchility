import abc


class MetricBase(abc.ABC):
    def __init__(self, on_step=None, on_epoch=True, name=None):
        self.log_step = on_step
        self.log_epoch = on_epoch
        self.name = self.__class__.__name__ if name is None else name

    def __call__(self, preds, targets):
        return self.forward(preds, targets)

    @abc.abstractmethod
    def forward(self, preds, targets):
        return NotImplemented
