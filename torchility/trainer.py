import torch
import inspect
from typing import Callable, Union
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ProgressBarBase

from torchmetrics import Metric

from .callbacks import ResetMetrics, SimpleBar
from .metrics import MetricBase
from .tasks import GeneralTaskModule
from .callbacks import Progress


def default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

class Trainer(PLTrainer):
    def __init__(self, model:torch.nn.Module=None,  # pytorch Module
        loss:Callable=None,                         # loss function
        optimizer:torch.optim.Optimizer=None,       # pytorch optimizer
        epochs=None,                                # max epochs
        metrics:Union[Callable, Metric]=(),         # instance of torchmetrics.Metric or other callable instance
        task_module: LightningModule=None,          # task_model
        datamodule: LightningDataModule = None,     # PL data module
        task_kwargs=dict(),                         # parameters of the task_module
        **pltrainer_kwargs                          # keyword arguments of pytorch_lightning Trainer
        ):

        self.init_params = default_args(PLTrainer)
        self.progress = None

        #   *************************************
        #   *    Task Configuration    --- LIC  *
        #   *************************************
        metrics = self._prepare_metrics(metrics)
        if task_module is None:
            self.task_module = GeneralTaskModule(model, loss, optimizer, metrics, **task_kwargs)
        else:
            self.task_module = task_module
        self.datamodule = datamodule

        #   *************************************
        #   *    Trainer Parameters    --- LIC  *
        #   *************************************
        if epochs is not None:  # for easy use
            self.init_params['max_epochs'] = epochs
        if pltrainer_kwargs.get('log_every_n_steps', None) is None: # log each step
            pltrainer_kwargs['log_every_n_steps'] = 1
        self.init_params.update(pltrainer_kwargs)     # get default arguments
        self.init_params['num_sanity_val_steps'] = 0  # how many validation steps to execute before running

        # === set callbacks
        cbks = [ResetMetrics()]
        if self.init_params['callbacks'] is not None:
            cbks.extend(self.init_params['callbacks'])
        if not any([isinstance(cbk, Progress) for cbk in cbks]):
            cbks.append(Progress())
        if not any([isinstance(cbk, ProgressBarBase) for cbk in cbks]):
            cbks.append(SimpleBar())  # ResetMetrics must stay before SimpleBar
        self.init_params['callbacks'] = cbks

        # === default logger
        if self.init_params['logger'] == True:
            if self.init_params['default_root_dir'] is None:
                log_dir = 'logs' 
            else:
                log_dir = self.init_params['default_root_dir']
            self.init_params['logger'] = TensorBoardLogger(log_dir, name=None, log_graph=True, default_hp_metric=False)

        super().__init__(**self.init_params)

    def _prepare_metrics(self, metrics):
        metrics_ready = {'train': [], 'val':[], 'test':[]}
        for m in metrics:
            if isinstance(m, (tuple, list)):  # when m is (metric_name, metric)
                assert len(m) == 2, '`metric` should be a tuple of (metric_name, metric_callable)'
                name, m = m
                m.name = name
            if isinstance(m, (Metric, MetricBase)):
                if not hasattr(m, 'name'):
                    m.name = m.__class__.__name__
                metrics_ready['train'].append(m)
                metrics_ready['val'].append(m.clone())
                metrics_ready['test'].append(m.clone())
            else:
                if not hasattr(m, 'name'):
                    name = m.__name__ if hasattr(m, '__name__') else type(m).__name__
                metrics_ready['train'].append(MetricBase(m, name))
                metrics_ready['val'].append(MetricBase(m, name))
                metrics_ready['test'].append(MetricBase(m, name))
        return metrics_ready

    def fit(self, train_dl=None, val_dl=None, epochs=None, ckpt_path=None):
        if self.datamodule:
            super().fit(self.task_module, datamodule=self.datamodule, ckpt_path=ckpt_path)
        else:
            super().fit(self.task_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt_path)
        return self.progress

    def test(self, test_dl=None, ckpt_path='best', pl_module=None, verbose=True):
        if test_dl is not None:
            return super().test(pl_module, dataloaders=test_dl, ckpt_path=ckpt_path, verbose=verbose)
        elif self.datamodule and self.datamodule.test_dataloader():
            return super().test(pl_module, datamodule=self.datamodule, ckpt_path=ckpt_path, verbose=verbose)
        else:
            raise Exception("Dataloader or DataModule is needed!")

    def resume_checkpoint(self, ckpt_path):
        """
        从checkpoint恢复trainer，然后可继续训练。
        注意，再次训练时epochs参数包含已训练的epoches。

        Note:
        ``resume_from_checkpoint`` is deprecated in pytorch-lightning v1.5 and will be removed in v1.7.
        Please pass the path to ``Trainer.fit(..., ckpt_path=...)`` instead.
        """
        self.init_params['resume_from_checkpoint'] = ckpt_path
        ckpt_trainer = Trainer(**self.init_params)
        ckpt_trainer.task_module = self.task_module
        ckpt_trainer.datamodule = self.datamodule
        return ckpt_trainer
