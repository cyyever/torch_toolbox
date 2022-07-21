import os

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.hook import Hook


class ModelExecutorLogger(Hook):
    def __init__(self):
        super().__init__(stripable=True)

    def _before_execute(self, **kwargs):
        model_executor = kwargs["model_executor"]
        model_util = model_executor.model_util
        if os.getenv("draw_torch_model") is not None:
            model_executor._model_with_loss.trace_input = True
        get_logger().info("dataset is %s", model_executor.dataset)
        get_logger().info("model type is %s", model_executor.model.__class__)
        get_logger().debug("model is %s", model_executor.model)
        get_logger().debug("loss function is %s", model_executor.loss_fun)
        get_logger().info(
            "parameter number is %s",
            len(model_util.get_parameter_list()),
        )
        get_logger().info("hyper_parameter is %s", model_executor.hyper_parameter)
        optimizer = model_executor.get_optimizer()
        if optimizer is not None:
            get_logger().info("optimizer is %s", optimizer)
        lr_scheduler = model_executor.get_lr_scheduler()
        if lr_scheduler is not None:
            get_logger().info(
                "lr_scheduler is %s",
                type(lr_scheduler),
            )

    def _after_execute(self, **kwargs):
        model_executor = kwargs["model_executor"]
        if os.getenv("draw_torch_model") is not None:
            model_executor.visualizer.writer.add_graph(
                model_executor.model, model_executor._model_with_loss.example_input
            )
