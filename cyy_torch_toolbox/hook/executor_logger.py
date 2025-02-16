from cyy_naive_lib.log import log_debug, log_info

from ..ml_type import MachineLearningPhase
from . import Hook


class ExecutorLogger(Hook):
    def __init__(self) -> None:
        super().__init__(stripable=True)

    def _before_execute(self, executor, **kwargs) -> None:
        log_info("dataset is %s", executor.dataset_collection.name)
        log_info("device is %s", executor.device)
        log_info("model type is %s", executor.model.__class__)
        log_debug("model is %s", executor.model)
        # log_debug("loss function is %s", executor.loss_fun)
        log_info(
            "parameter number is %s",
            sum(a.numel() for a in executor.model_util.get_parameter_seq(detach=False)),
        )
        trainable_parameter_number = 0
        dtype_stat = {}
        device_stat = {}
        for name, parameter in executor.model.named_parameters():
            if parameter.requires_grad:
                trainable_parameter_number += parameter.numel()
            if parameter.device not in device_stat:
                device_stat[parameter.device] = []
            device_stat[parameter.device].append(name)
            if parameter.dtype not in dtype_stat:
                dtype_stat[parameter.dtype] = []
            dtype_stat[parameter.dtype].append(name)
        log_info("trainable parameter number is %s", trainable_parameter_number)
        if len(device_stat) == 1:
            log_info("model use device %s", list(device_stat.keys())[0])
        else:
            log_info("model use device %s", device_stat)
        log_info("model use dtype %s", list(dtype_stat.keys()))
        if hasattr(executor, "hyper_parameter"):
            log_info("hyper_parameter is %s", executor.hyper_parameter)
        if executor.has_optimizer():
            log_info("optimizer is %s", executor.get_optimizer())
        if executor.has_lr_scheduler():
            log_info("lr_scheduler is %s", type(executor.get_lr_scheduler()))
        for phase in MachineLearningPhase:
            if executor.dataset_collection.has_dataset(phase):
                log_info(
                    "%s dataset size %s",
                    phase,
                    len(executor.dataset_collection.get_dataset_util(phase=phase)),
                )
