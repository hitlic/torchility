import os
from copy import deepcopy
import torch
import inspect
from typing import Callable, Union
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ProgressBar, ModelCheckpoint

from torchmetrics import Metric

from .callbacks import ResetMetrics, SimpleBar
from .tasks import GeneralTaskModule
from .callbacks import Progress, LogHyperParameters
from .utils import batches, set_metric_attr, load_state_dict
from .datamodule import GeneralDataModule

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
        epochs:int=None,                            # max epochs
        metrics:Union[Callable, Metric]=(),         # instance of torchmetrics.Metric or other callable instance
        task_module: LightningModule=None,          # task_model
        default_bar=False,                          # 是否使用默认进度条
        reset_dl:int=0,                             # 每隔多少个epoch重置一次训练DataLoader，与reload_dataloaders_every_n_epochs相似
        val_freq=1,                                 # 每隔多少个epoch验证一次，与check_val_every_n_epoch相同
        long_output=False,                          # 进度输出中保留7位（True）还是4位（False）小数。
        hyper_parameters=None,                      # 需要被日志记录的超参数（用于调参）
        task_kwargs:dict=None,                      # parameters dict of the task_module
        **pltrainer_kwargs                          # keyword arguments of pytorch_lightning Trainer
        ):
        self.init_params = default_args(PLTrainer)
        self.progress = None
        self.best_model = None   # 训练中最好的torch模型

        #   *************************************
        #   *    Task Configuration    --- LIC  *
        #   *************************************

        metrics = self._prepare_metrics(metrics)
        if task_module is None:
            task_kwargs = task_kwargs if task_kwargs is not None else {}
            self.task_module = GeneralTaskModule(model, loss, optimizer, metrics, **task_kwargs)
        else:
            self.task_module = task_module
        self.datamodule = None

        #   *************************************
        #   *    Trainer Parameters    --- LIC  *
        #   *************************************
        if epochs is not None:  # for easy use
            self.init_params['max_epochs'] = epochs
        if pltrainer_kwargs.get('log_every_n_steps', None) is None: # log each step
            pltrainer_kwargs['log_every_n_steps'] = 1
        if pltrainer_kwargs.get('check_val_every_n_epoch', None) is None:  # val freq
            pltrainer_kwargs['check_val_every_n_epoch'] = val_freq
        self.init_params.update(pltrainer_kwargs)     # get default arguments
        self.init_params['num_sanity_val_steps'] = 0  # how many validation steps to execute before running

        # === set callbacks
        cbks = [ResetMetrics()]
        if self.init_params['callbacks'] is not None:
            cbks.extend(self.init_params['callbacks'])
        if self.init_params['enable_progress_bar'] !=False and not default_bar:
            if not any(isinstance(cbk, Progress) for cbk in cbks):
                cbks.append(Progress())
            if not any(isinstance(cbk, ProgressBar) for cbk in cbks):
                cbks.append(SimpleBar(long_output))  # ResetMetrics must stay before SimpleBar

        # log hyperparemeters
        if hyper_parameters is not None:
            if not isinstance(hyper_parameters, dict):
                raise ValueError('`hyper_parameters` must be a dict!')
            cbks.append(LogHyperParameters(hyper_parameters))

        # checkpoint
        self._checkpoint = None
        enable_cpk = self.init_params['enable_checkpointing']
        if enable_cpk is None or enable_cpk:
            if not any(isinstance(cbk, ModelCheckpoint) for cbk in cbks):
                self._checkpoint = ModelCheckpoint()
                cbks.append(self._checkpoint)

        self.init_params['callbacks'] = cbks

        # === default logger
        if self.init_params['logger'] is None or self.init_params['logger'] == True:
            if self.init_params['default_root_dir'] is None:
                log_dir = 'torchility_logs'
            else:
                log_dir = self.init_params['default_root_dir']
            self.init_params['logger'] = TensorBoardLogger(log_dir, name=None, log_graph=True, default_hp_metric=False)

        # === reset dataloader
        self.reset_dl = reset_dl
        if reset_dl > 0:
            self.init_params['reload_dataloaders_every_n_epochs'] = reset_dl

        self.val_epoch_metric_values = {}
        self.train_epoch_metric_values = {}
        self.test_epoch_metric_values = {}

        super().__init__(**self.init_params)

    def _prepare_metrics(self, metrics, just_for_test=False):
        metrics_ready = {'train': [], 'val':[], 'test':[]}
        for m in metrics:
            m = set_metric_attr(m)
            if just_for_test:
                metrics_ready['test'].append(m)
            else:
                for stage in m.stages:
                    metrics_ready[stage].append(m.clone())
        del metrics
        return metrics_ready if not just_for_test else metrics_ready['test']

    def fit(self, train_dl, val_dls=None, ckpt_path=None, do_val_loss=True):
        """
        Args:
            train_dl: 用于训练的dataloader
            val_dls:  用于验证的一个dataloader或多个dataloader列表
            ckpt_path: checkpoints保存路径
            do_val_loss: True表示在验证时计算损失；
                         False表示验证时不计算损失；
                         int 表示在使用多个验证dataloader时，在哪个dataloader上计算损失
                         [int, ...]整数列表，表示在在哪些dataloader上计算损失
        """
        # 如果有验证集且modelcheckpoint没有指定monitor，则以val_loss为monitor
        if val_dls is not None and self._checkpoint and self._checkpoint.monitor is None:
            self._checkpoint.monitor = 'val_loss'

        if isinstance(do_val_loss, (list, tuple)):
            assert isinstance(val_dls, (list, tuple)), 'do_val_loss 的取值为val_dls中dataloader对应的id'
            assert len(do_val_loss) < len(val_dls), 'do_val_loss 的取值为val_dls中dataloader对应的id'
        elif do_val_loss.__class__ is int:
            do_val_loss = [do_val_loss]
        self.task_module.do_val_loss = do_val_loss

        self.task_module.multi_val_dataloaders = False  # 是否使用多个验证dataloader
        if isinstance(val_dls, (list, tuple)) and len(val_dls) > 1:
            self.task_module.multi_val_dataloaders = True

        dm = GeneralDataModule(train_dl=train_dl, val_dls=val_dls, reset=self.reset_dl)
        self.datamodule = dm
        super().fit(self.task_module, datamodule=dm, ckpt_path=ckpt_path)
        return self.progress

    def test(self, test_dls, ckpt_path='best', verbose=True, metrics=None, do_loss=True):  # pylint: disable=arguments-renamed
        """
        Args:
            metrics: 一旦提供了test的metrics，则会将trainer中用于test的metrics覆盖，也就是说可以指定仅用于test的metrics
            do_loss: True表示在测试时计算损失；
                     False表示测试时不计算损失；
                     [int, ...]整数列表示在使用多个测试dataloader时，在哪些dataloader上计算损失
        """
        print('-' * 35)
        if self.init_params['enable_checkpointing'] == False:  # 当没有checkpoint时
            print("NOTICE: Using the latest model for Test!")
            ckpt_path = None

        if isinstance(do_loss, (list, tuple)):
            assert isinstance(test_dls, (list, tuple)), 'do_loss 的取值为test_dls中dataloader对应的id'
            assert len(do_loss) < len(test_dls), 'do_loss 的取值为test_dls中dataloader对应的id'
        elif do_loss.__class__ is int:
            do_loss = [do_loss]
        self.task_module.do_test_loss = do_loss

        self.task_module.multi_test_dataloaders = False          # 是否使用多个测试dataloader
        if isinstance(test_dls, (list, tuple)) and len(test_dls) > 1:
            self.task_module.multi_test_dataloaders = True

        if metrics is not None:  # 如果提供了仅用于test的metrics
            test_metrics = self._prepare_metrics(metrics, just_for_test=True)
            self.task_module.metrics['test'] = test_metrics

        dm = GeneralDataModule(test_dls=test_dls)
        super().test(self.task_module, datamodule=dm, ckpt_path=ckpt_path, verbose=False)

        if verbose:
            for k, v in self.test_metrics.items():
                print(f'{k}:\t {v}')
        print('-' * 35, '\n')
        return self.test_metrics

    def predict(self, pred_dl, has_label=False, ckpt_path='best', concat=True):  # pylint: disable=arguments-renamed
        self.task_module.pred_dataloader_has_label = has_label
        dm = GeneralDataModule(pred_dl=pred_dl)
        preds = super().predict(self.task_module, datamodule=dm, return_predictions=True, ckpt_path=ckpt_path)
        if concat:
            return torch.cat(preds, dim=0)
        else:
            return preds

    def predict_with_best(self, *inputs, batch_size=0, train=False):
        """
        利用最佳模型对inputs进行预测，返回预测结果
        Args:
            inputs: Tensor或者Tensor列表
            batch_size: 如果batch_size == 0则不划分minibatch
            train: 模型以训练模式(train)还是评价模式(eval)进行预测
        """
        if self.best_model is None:
            self.best_model = self.restore_best()
        if train:
            self.best_model.train()
        else:
            self.best_model.eval()
        device = next(self.best_model.parameters()).device

        if batch_size > 0:
            preds = []
            for batch in batches(inputs, batch_size):
                batch = batch.to(device)
                with torch.no_grad():
                    pred_out = self.best_model(*batch)
                preds.append(pred_out)
            return torch.cat(preds)
        else:
            with torch.no_grad():
                inputs = [data.to(device) for data in inputs]
                pred_out = self.best_model(*inputs)
            return pred_out

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
        restore best k checkpoints. k is the value of `save_top_k` argument in the `ModelCheckpoint` callback.
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

    def save_torch_state_dict(self, path='best_model.pth', mode='best'):
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

    def load_torch_state_dict(self, path='best_model.pth', model=None):
        """load state_dict for pytorch model"""
        if model:
            model.load_state_dict(torch.load(path))
        else:
            self.task_module.model.load_state_dict(torch.load(path))
            model = self.task_module.model
        return model

    @staticmethod
    def load_state_dict(model, ckpt_path):
        """
        load state_dict for pytorch model from pytorch_lightning checkpoint.
        Args:
            model: pytorch model
            ckpt_path: path of pytorch_lightning checkpoint
        """
        return load_state_dict(model, ckpt_path)

    def run_tensorboard(self, logdir=None):
        if logdir is None:
            if self.logger is None:
                logdir = os.path.dirname(os.path.realpath(__file__))
            else:
                logdir = self.log_dir.rsplit('/', 1)[0]
        print(f'logdir is {logdir}')
        os.system(f'tensorboard --logdir={logdir}')
