from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import MachineLearningPhase

# import os


class ExecutorLogger(Hook):
    def __init__(self):
        super().__init__(stripable=True)

    def _before_execute(self, executor, **kwargs):
        model_util = executor.model_util
        get_logger().info(
            "dataset type is %s",
            executor.dataset_collection.get_original_dataset(
                MachineLearningPhase.Training
            ),
        )
        get_logger().info("model type is %s", executor.model.__class__)
        get_logger().debug("model is %s", executor.model)
        get_logger().debug("loss function is %s", executor.loss_fun)
        get_logger().info(
            "parameter number is %s",
            len(model_util.get_parameter_list()),
        )
        if hasattr(executor, "hyper_parameter"):
            get_logger().info("hyper_parameter is %s", executor.hyper_parameter)
        optimizer = executor.get_optimizer()
        if optimizer is not None:
            get_logger().info("optimizer is %s", optimizer)
        lr_scheduler = executor.get_lr_scheduler()
        if lr_scheduler is not None:
            get_logger().info("lr_scheduler is %s", type(lr_scheduler))
        for phase in MachineLearningPhase:
            if executor.dataset_collection.has_dataset(phase):
                get_logger().info("%s dataset len %s", phase, executor.dataset_size)
        if executor.dataset_collection.tokenizer is not None:
            get_logger().info(
                "tokenizer is %s", type(executor.dataset_collection.tokenizer)
            )
        # if os.getenv("draw_torch_model") is not None:
        #     executor._model_with_loss.trace_input = True

    # def _after_execute(self, **kwargs):
    #     executor = kwargs["executor"]
    #     if os.getenv("draw_torch_model") is not None:
    #         executor.visualizer.writer.add_graph(
    #             executor.model, executor._model_with_loss.example_input
    #         )
