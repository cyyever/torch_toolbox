import datetime
import os
import threading
import warnings

warnings.simplefilter("ignore", DeprecationWarning)

# from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from torch.utils.tensorboard import SummaryWriter

from .metric_visualizer import MetricVisualizer


class MetricTensorBoard(MetricVisualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__writer = None

    def close(self) -> None:
        if self.__writer is not None:
            self.__writer.close()
            self.__writer = None

    def set_prefix(self, prefix: str) -> None:
        super().set_prefix(prefix)
        self.close()

    @property
    def writer(self) -> SummaryWriter:
        if self.__writer is None:
            self.__writer = SummaryWriter(
                log_dir=os.path.join(self._data_dir, self.prefix)
            )
        return self.__writer

    def _before_execute(self, executor, **kwargs):
        if self.prefix is None:
            self.set_prefix(
                "training_"
                + str(executor.model.__class__.__name__)
                + "_"
                + str(threading.get_native_id())
                + "_{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.datetime.now())
            )

    def get_tag_name(self, title: str) -> str:
        return self.prefix + "/" + title

    def __del__(self):
        self.close()

    def _after_validation(self, executor, epoch, **kwargs):
        trainer = executor

        # if "cur_learning_rates" not in trainer._data:
        #     return
        # learning_rates = trainer._data["cur_learning_rates"]
        # assert len(learning_rates) == 1
        # self.writer.add_scalar(
        #     self.get_tag_name("learning rate"), learning_rates[0], epoch
        # )
        optimizer = trainer.get_optimizer()
        for group in optimizer.param_groups:
            if "momentum" in group:
                momentum = group["momentum"]
                self.writer.add_scalar(self.get_tag_name("momentum"), momentum, epoch)

        validation_metric = trainer.get_cached_inferencer(
            MachineLearningPhase.Validation
        ).get_hook("performance_metric")
        performance_metric = trainer.get_hook("performance_metric")
        self.writer.add_scalars(
            self.get_tag_name("training & validation loss"),
            {
                "training loss": performance_metric.get_loss(epoch),
                "validation loss": validation_metric.get_loss(epoch),
            },
            epoch,
        )

        self.writer.add_scalars(
            self.get_tag_name("training & validation accuracy"),
            {
                "training accuracy": performance_metric.get_accuracy(epoch),
                "validation accuracy": validation_metric.get_accuracy(epoch),
            },
            epoch,
        )

        # if trainer.has_data("plot_class_accuracy"):
        #     class_accuracy = validation_metric.get_class_accuracy(epoch)
        #     for idx, sub_list in enumerate(
        #         split_list_to_chunks(list(class_accuracy.keys()), 2)
        #     ):
        #         class_accuracy_title = "validation class accuracy part " + str(idx)
        #         for k in sub_list:
        #             self.writer.add_scalars(
        #                 self.get_tag_name(class_accuracy_title),
        #                 {
        #                     "class_" + str(k) + "_accuracy": class_accuracy[k],
        #                 },
        #                 epoch,
        #             )
        test_metric = trainer.get_cached_inferencer(MachineLearningPhase.Test).get_hook(
            "performance_metric"
        )
        self.writer.add_scalar(
            self.get_tag_name("test loss"),
            test_metric.get_epoch_metric(epoch, "loss"),
            epoch,
        )
        self.writer.add_scalar(
            self.get_tag_name("test accuracy"),
            test_metric.get_epoch_metric(epoch, "accuracy"),
            epoch,
        )
        self.writer.flush()
