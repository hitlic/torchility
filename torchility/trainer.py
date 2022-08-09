from copy import deepcopy
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

    def _prepare_metrics(self, metrics, just_for_test=False):
        metrics_ready = {'train': [], 'val':[], 'test':[]}
        for m in metrics:
            if isinstance(m, (tuple, list)):  # when m is (metric_name, metric)
                assert len(m) == 2, '`metric` should be a tuple of (metric_name, metric_callable)'
                name, m = m
                m.name = name
            if isinstance(m, (Metric, MetricBase)):
                if not hasattr(m, 'name'):
                    m.name = m.__class__.__name__
                if just_for_test:
                    metrics_ready['test'].append(m.clone())
                else:
                    metrics_ready['train'].append(m)
                    metrics_ready['val'].append(m.clone())
                    metrics_ready['test'].append(m.clone())
            else:
                if not hasattr(m, 'name'):
                    name = m.__name__ if hasattr(m, '__name__') else type(m).__name__
                else:
                    name = m.name
                if just_for_test:
                    metrics_ready['test'].append(MetricBase(m, name))
                else:
                    metrics_ready['train'].append(MetricBase(m, name))
                    metrics_ready['val'].append(MetricBase(m, name))
                    metrics_ready['test'].append(MetricBase(m, name))
        return metrics_ready if not just_for_test else metrics_ready['test']

    def fit(self, train_dl=None, val_dl=None, epochs=None, ckpt_path=None):
        if self.datamodule:
            super().fit(self.task_module, datamodule=self.datamodule, ckpt_path=ckpt_path)
        else:
            super().fit(self.task_module, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt_path)
        return self.progress

    def test(self, test_dl=None, ckpt_path='best', pl_module=None, verbose=True, metrics=None, do_loss=True):
        """
        metrics: 一旦提供了test的metrics，则会将trainer中用于test的metrics覆盖，也就是说可以指定仅用于test的metrics
        do_loss: 在测试时是否计算损失函数，当测试任务和训练任务数据不一致无法计算损失函数时，可设do_loss=False
        """
        self.task_module.do_test_loss = do_loss
        if metrics is not None:  # 如果提供了仅用于test的metrics
            test_metrics = self._prepare_metrics(metrics, just_for_test=True)
            self.task_module.metrics['test'] = test_metrics

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

    def restore_best(self):
        """
        restore best checkpoint
        """
        model = self.task_module.model
        loss = self.task_module.loss_fn
        opt = self.task_module.opt
        self.task_module = self.task_module.load_from_checkpoint(self.checkpoint_callback.best_model_path, 
                                                            model=model, loss=loss, optimizer=opt)
        return self.task_module.model

    def restore_best_k(self):
        """
        restore best k checkpoints. k is value of `save_top_k` argument in the `ModelCheckpoint` callback.
        """
        best_ckp_paths = self.checkpoint_callback.best_k_models.keys()
        best_models = []
        loss = self.task_module.loss_fn
        opt = self.task_module.opt
        model =self.task_module.model
        for ckp_path in best_ckp_paths:
            self.task_module.load_from_checkpoint(ckp_path, model=model, loss=loss, optimizer=opt)
            best_models.append(deepcopy(self.task_module.model))
        return best_models


    def save_state_dict(self, path='best_model.pth', mode='best'):
        """
        Save state_dict of pytorch model.
        Args:
            path: path and filename of the state_dict file.
            mode: 'best' for the best model and others for current model.
        """
        if mode == 'best':
            model = self.task_module.model
            loss = self.task_module.loss_fn
            opt = self.task_module.opt
            task = self.task_module.load_from_checkpoint(self.checkpoint_callback.best_model_path, 
                                                            model=model, loss=loss, optimizer=opt)
            torch.save(task.model.state_dict(), path)
        else:
            torch.save(self.task_module.model.state_dict(), path)

    def load_state_dict(self, path='best_model.pth', model=None):
        """load state_dict for pytorch model"""
        if model:
            model.load_state_dict(torch.load(path))
        else:
            self.task_module.model.load_state_dict(torch.load(path))
            model = self.task_module.model
        return model
