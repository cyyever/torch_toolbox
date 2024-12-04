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
        log_info(
            "trainable parameter number is %s",
            sum(a.numel() for a in executor.model_util.get_parameter_seq(detach=False) if a.requires_grad),
        )
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
