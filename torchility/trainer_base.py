from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.plugins import Plugin
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.profiler import BaseProfiler
from pytorch_lightning.trainer.connectors.env_vars_connector import _defaults_from_env_vars
from pytorch_lightning.loggers import TensorBoardLogger

class TrainerBase(PLTrainer):
    @_defaults_from_env_vars
    def __init__(
        self,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        checkpoint_callback: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = 'norm',
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        log_gpu_memory: Optional[str] = None,
        progress_bar_refresh_rate: Optional[int] = None,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        limit_predict_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        sync_batchnorm: bool = False,
        precision: int = 32,
        weights_summary: Optional[str] = 'top',
        weights_save_path: Optional[str] = None,
        num_sanity_val_steps: int = 0,                          # 训练开始前预验证的step数 
        truncated_bptt_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[BaseProfiler, str]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        terminate_on_nan: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        plugins: Optional[Union[List[Union[Plugin, ClusterEnvironment, str]], Plugin, ClusterEnvironment, str]] = None,
        amp_backend: str = 'native',
        amp_level: str = 'O2',
        distributed_backend: Optional[str] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = 'max_size_cycle',
        stochastic_weight_avg: bool = False
    ):
        if logger==True:
            logger = TensorBoardLogger('logs', name=None, log_graph=True, default_hp_metric=False)
        super().__init__(logger=logger, checkpoint_callback=checkpoint_callback, callbacks=callbacks, default_root_dir=default_root_dir, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, process_position=process_position, num_nodes=num_nodes, num_processes=num_processes, gpus=gpus, auto_select_gpus=auto_select_gpus, tpu_cores=tpu_cores, log_gpu_memory=log_gpu_memory, progress_bar_refresh_rate=progress_bar_refresh_rate, overfit_batches=overfit_batches, track_grad_norm=track_grad_norm, check_val_every_n_epoch=check_val_every_n_epoch, fast_dev_run=fast_dev_run, accumulate_grad_batches=accumulate_grad_batches, max_epochs=max_epochs, min_epochs=min_epochs, max_steps=max_steps, min_steps=min_steps, max_time=max_time, limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches, limit_test_batches=limit_test_batches, limit_predict_batches=limit_predict_batches, val_check_interval=val_check_interval,
                         flush_logs_every_n_steps=flush_logs_every_n_steps, log_every_n_steps=log_every_n_steps, accelerator=accelerator, sync_batchnorm=sync_batchnorm, precision=precision, weights_summary=weights_summary, weights_save_path=weights_save_path, num_sanity_val_steps=num_sanity_val_steps, truncated_bptt_steps=truncated_bptt_steps, resume_from_checkpoint=resume_from_checkpoint, profiler=profiler, benchmark=benchmark, deterministic=deterministic, reload_dataloaders_every_epoch=reload_dataloaders_every_epoch, auto_lr_find=auto_lr_find, replace_sampler_ddp=replace_sampler_ddp, terminate_on_nan=terminate_on_nan, auto_scale_batch_size=auto_scale_batch_size, prepare_data_per_node=prepare_data_per_node, plugins=plugins, amp_backend=amp_backend, amp_level=amp_level, distributed_backend=distributed_backend, move_metrics_to_cpu=move_metrics_to_cpu, multiple_trainloader_mode=multiple_trainloader_mode, stochastic_weight_avg=stochastic_weight_avg)
        self.init_params = dict(logger=logger, checkpoint_callback=checkpoint_callback, callbacks=callbacks, default_root_dir=default_root_dir, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, process_position=process_position, num_nodes=num_nodes, num_processes=num_processes, gpus=gpus, auto_select_gpus=auto_select_gpus, tpu_cores=tpu_cores, log_gpu_memory=log_gpu_memory, progress_bar_refresh_rate=progress_bar_refresh_rate, overfit_batches=overfit_batches, track_grad_norm=track_grad_norm, check_val_every_n_epoch=check_val_every_n_epoch, fast_dev_run=fast_dev_run, accumulate_grad_batches=accumulate_grad_batches, max_epochs=max_epochs, min_epochs=min_epochs, max_steps=max_steps, min_steps=min_steps, max_time=max_time, limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches, limit_test_batches=limit_test_batches, limit_predict_batches=limit_predict_batches, val_check_interval=val_check_interval,
                                flush_logs_every_n_steps=flush_logs_every_n_steps, log_every_n_steps=log_every_n_steps, accelerator=accelerator, sync_batchnorm=sync_batchnorm, precision=precision, weights_summary=weights_summary, weights_save_path=weights_save_path, num_sanity_val_steps=num_sanity_val_steps, truncated_bptt_steps=truncated_bptt_steps, resume_from_checkpoint=resume_from_checkpoint, profiler=profiler, benchmark=benchmark, deterministic=deterministic, reload_dataloaders_every_epoch=reload_dataloaders_every_epoch, auto_lr_find=auto_lr_find, replace_sampler_ddp=replace_sampler_ddp, terminate_on_nan=terminate_on_nan, auto_scale_batch_size=auto_scale_batch_size, prepare_data_per_node=prepare_data_per_node, plugins=plugins, amp_backend=amp_backend, amp_level=amp_level, distributed_backend=distributed_backend, move_metrics_to_cpu=move_metrics_to_cpu, multiple_trainloader_mode=multiple_trainloader_mode, stochastic_weight_avg=stochastic_weight_avg)
