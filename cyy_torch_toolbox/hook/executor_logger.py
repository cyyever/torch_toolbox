from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import MachineLearningPhase

# import os


class ExecutorLogger(Hook):
    def __init__(self) -> None:
        super().__init__(stripable=True)

    def _before_execute(self, executor, **kwargs) -> None:
        get_logger().info(
            "dataset type is %s",
            executor.dataset_collection.get_original_dataset(
                MachineLearningPhase.Training
            ),
        )
        get_logger().info("model type is %s", executor.model.__class__)
        get_logger().debug("model is %s", executor.model)
        # get_logger().debug("loss function is %s", executor.loss_fun)
        get_logger().info(
            "parameter number is %s",
            sum(a.numel() for a in executor.model_util.get_parameter_seq()),
        )
        if hasattr(executor, "hyper_parameter"):
            get_logger().info("hyper_parameter is %s", executor.hyper_parameter)
        if executor.has_optimizer():
            get_logger().info("optimizer is %s", executor.get_optimizer())
        if executor.has_lr_scheduler():
            get_logger().info("lr_scheduler is %s", type(executor.get_lr_scheduler()))
        for phase in MachineLearningPhase:
            if executor.dataset_collection.has_dataset(phase):
                get_logger().info(
                    "%s dataset len %s",
                    phase,
                    len(executor.dataset_collection.get_dataset_util(phase=phase)),
                )
