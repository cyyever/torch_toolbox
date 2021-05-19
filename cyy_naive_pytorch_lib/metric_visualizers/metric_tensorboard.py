import datetime
import threading

from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from torch.utils.tensorboard import SummaryWriter

from ml_type import MachineLearningPhase

from .metric_visualizer import MetricVisualizer


class MetricTensorBoard(MetricVisualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__writer = None
        self.__log_dir = None

    def set_log_dir(self, log_dir: str):
        self.__log_dir = log_dir

    def close(self):
        if self.__writer is not None:
            self.__writer.close()
            self.__writer = None

    def set_session_name(self, name: str):
        super().set_session_name(name)
        self.close()

    def _before_execute(self, **kwargs):
        trainer = kwargs["model_executor"]
        if self.session_name is None:
            self.set_session_name(
                "training_"
                + str(trainer.model.__class__.__name__)
                + "_"
                + str(threading.get_native_id())
                + "_{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
            )

    def get_tag_name(self, title: str):
        return self.session_name + "/" + title

    def __del__(self):
        self.close()

    @property
    def writer(self):
        if self.__writer is None:
            assert self.session_name
            self.__writer = SummaryWriter(self.__log_dir + "/" + self.session_name)
        return self.__writer

    def _after_epoch(self, **kwargs):
        trainer = kwargs["model_executor"]
        epoch = kwargs["epoch"]
        learning_rates = trainer.get_data("cur_learning_rates")
        assert len(learning_rates) == 1
        self.writer.add_scalar(
            self.get_tag_name("learning rate"), learning_rates[0], epoch
        )
        optimizer = trainer.get_optimizer()
        for group in optimizer.param_groups:
            if "momentum" in group:
                momentum = group["momentum"]
                self.writer.add_scalar(self.get_tag_name("momentum"), momentum, epoch)

        validation_metric = trainer.get_inferencer_performance_metric(
            MachineLearningPhase.Validation
        )
        self.writer.add_scalars(
            self.get_tag_name("training & validation loss"),
            {
                "training loss": trainer.performance_metric.get_loss(epoch),
                "validation loss": validation_metric.get_loss(epoch),
            },
            epoch,
        )

        self.writer.add_scalars(
            self.get_tag_name("training & validation accuracy"),
            {
                "training accuracy": trainer.performance_metric.get_accuracy(epoch),
                "validation accuracy": validation_metric.get_accuracy(epoch),
            },
            epoch,
        )

        if trainer.has_data("plot_class_accuracy"):
            class_accuracy = validation_metric.get_class_accuracy(epoch)
            for idx, sub_list in enumerate(
                split_list_to_chunks(list(class_accuracy.keys()), 2)
            ):
                class_accuracy_title = "validation class accuracy part " + str(idx)
                for k in sub_list:
                    self.writer.add_scalars(
                        self.get_tag_name(class_accuracy_title),
                        {
                            "class_" + str(k) + "_accuracy": class_accuracy[k],
                        },
                        epoch,
                    )
        test_metric = trainer.get_inferencer_performance_metric(
            MachineLearningPhase.Test
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
