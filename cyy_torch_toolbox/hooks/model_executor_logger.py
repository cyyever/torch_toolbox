import os

from cyy_naive_lib.log import get_logger
from hook import Hook
from model_util import ModelUtil


class ModelExecutorLogger(Hook):
    def __init__(self):
        super().__init__(stripable=True)

    def _before_execute(self, **kwargs):
        model_executor = kwargs["model_executor"]
        model_util = ModelUtil(model_executor.model)
        if os.getenv("draw_torch_model") is not None:
            model_executor._model_with_loss.trace_input = True
        get_logger().info("dataset is %s", model_executor.dataset)
        get_logger().info("model type is %s", model_executor.model.__class__)
        get_logger().debug("model is %s", model_executor.model)
        get_logger().info("loss function is %s", model_executor.loss_fun)
        get_logger().info(
            "parameter number is %s",
            len(model_util.get_parameter_list()),
        )
        get_logger().info("use device %s", model_executor.device)
        get_logger().info("hyper_parameter is %s", model_executor.hyper_parameter)
        if hasattr(model_executor, "get_optimizer"):
            get_logger().info(
                "optimizer is %s", getattr(model_executor, "get_optimizer")()
            )
        if hasattr(model_executor, "get_lr_scheduler"):
            get_logger().info(
                "lr_scheduler is %s", getattr(model_executor, "get_lr_scheduler")()
            )

    def _after_execute(self, **kwargs):
        model_executor = kwargs["model_executor"]
        if os.getenv("draw_torch_model") is not None:
            model_executor.visualizer.writer.add_graph(
                model_executor.model, model_executor._model_with_loss.example_input
            )
