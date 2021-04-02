import datetime
import threading

from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger
from ml_type import MachineLearningPhase
from visualization import EpochWindow, Window

from .metric_visualizer import MetricVisualizer


class MetricVisdom(MetricVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__visdom_env = None

    def _before_execute(self, **kwargs):
        trainer = kwargs["model_executor"]
        self.__visdom_env = (
            "training_"
            + str(trainer.model.__class__.__name__)
            + "_"
            + str(threading.get_native_id())
            + "_{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
        )

    def _after_epoch(self, **kwargs):
        trainer = kwargs["model_executor"]
        epoch = kwargs["epoch"]
        learning_rates = trainer.get_data("cur_learning_rates")
        assert len(learning_rates) == 1
        EpochWindow("learning rate", env=self.__visdom_env).plot_learning_rate(
            epoch, learning_rates[0]
        )
        optimizer = trainer.get_optimizer()
        for group in optimizer.param_groups:
            if "momentum" in group:
                momentum = group["momentum"]
                momentum_win = EpochWindow("momentum", env=self.__visdom_env)
                momentum_win.y_label = "Momentum"
                momentum_win.plot_scalar(epoch, momentum)

        loss_win = EpochWindow("training & validation loss", env=self.__visdom_env)
        loss_win.plot_loss(epoch, trainer.loss_metric.get_loss(epoch), "training loss")

        validation_metric = trainer.get_validation_metric(
            MachineLearningPhase.Validation
        )

        loss_win = EpochWindow("training & validation loss", env=self.__visdom_env)
        loss_win.plot_loss(
            epoch, validation_metric.get_epoch_metric(epoch, "loss"), "validation loss"
        )
        EpochWindow("validation accuracy", env=self.__visdom_env).plot_accuracy(
            epoch,
            validation_metric.get_epoch_metric(epoch, "accuracy"),
        )

        if trainer.has_data("plot_class_accuracy"):
            class_accuracy = validation_metric.get_class_accuracy(epoch)
            for idx, sub_list in enumerate(
                split_list_to_chunks(list(class_accuracy.keys()), 2)
            ):
                class_accuracy_win = EpochWindow(
                    "class accuracy part " + str(idx), env=self.__visdom_env
                )
                for k in sub_list:
                    class_accuracy_win.plot_accuracy(
                        epoch,
                        class_accuracy[k],
                        "class_" + str(k) + "_accuracy",
                    )
        test_metric = trainer.get_validation_metric(MachineLearningPhase.Test)
        loss_win = EpochWindow("test loss", env=self.__visdom_env)
        loss_win.plot_loss(epoch, test_metric.get_epoch_metric(epoch, "loss"))
        EpochWindow("test accuracy", env=self.__visdom_env).plot_accuracy(
            epoch, test_metric.get_epoch_metric(epoch, "accuracy")
        )
        Window.save_envs()
