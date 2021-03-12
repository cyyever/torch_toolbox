import datetime
import threading

from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger

from basic_trainer import BasicTrainer
from dataset_collection import DatasetCollection
from hyper_parameter import HyperParameter
from ml_type import MachineLearningPhase
from model_executor import ModelExecutor, ModelExecutorCallbackPoint
from model_util import ModelUtil
from model_with_loss import ModelWithLoss
from visualization import EpochWindow, Window


class Trainer(BasicTrainer):
    """
    This trainer is designed to add logging to BasicTrainer
    """

    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset_collection: DatasetCollection,
        hyper_parameter: HyperParameter,
    ):
        super().__init__(
            model_with_loss=model_with_loss,
            dataset_collection=dataset_collection,
            hyper_parameter=hyper_parameter,
        )
        self.visdom_env = None
        self.add_callback(
            ModelExecutorCallbackPoint.BEFORE_EXECUTE, self.__pre_training_callback
        )
        self.add_callback(
            ModelExecutorCallbackPoint.AFTER_BATCH, Trainer.__log_after_batch
        )
        self.add_callback(
            ModelExecutorCallbackPoint.AFTER_EPOCH, Trainer.__plot_after_epoch
        )
        self.set_data("plot_class_accuracy", False)

    def enable_class_accuracy_plot(self):
        self.set_data("plot_class_accuracy", True)

    def __pre_training_callback(self, trainer):
        self.visdom_env = (
            "training_"
            + str(self.model.__class__.__name__)
            + "_"
            + str(threading.get_native_id())
            + "_{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
        )
        model_util = ModelUtil(trainer.model)
        get_logger().info(
            "begin training, hyper_parameter is %s, optimizer is %s ,lr_scheduler is %s, %s, parameter number is %s",
            trainer.hyper_parameter,
            trainer.get_optimizer(),
            trainer.get_lr_scheduler(),
            trainer.model_with_loss,
            len(model_util.get_parameter_list()),
        )

    @staticmethod
    def __log_after_batch(trainer: BasicTrainer, **kwargs):
        training_set_size = trainer.get_data("training_set_size")
        batch = kwargs["batch"]
        batch_index = kwargs["batch_index"]
        ten_batches = training_set_size // (10 * ModelExecutor.get_batch_size(batch[0]))
        if ten_batches == 0 or batch_index % ten_batches == 0:
            get_logger().info(
                "epoch: %s, batch: %s, learning rate: %s, batch training loss: %s",
                kwargs["epoch"],
                batch_index,
                trainer.get_data("cur_learning_rates"),
                kwargs["batch_loss"],
            )

    @staticmethod
    def __plot_after_epoch(trainer: BasicTrainer, epoch):
        learning_rates = trainer.get_data("cur_learning_rates")
        assert len(learning_rates) == 1
        EpochWindow("learning rate", env=trainer.visdom_env).plot_learning_rate(
            epoch, learning_rates[0]
        )
        optimizer = trainer.get_optimizer()
        for group in optimizer.param_groups:
            if "momentum" in group:
                momentum = group["momentum"]
                EpochWindow("momentum", env=trainer.visdom_env).plot_scalar(
                    epoch, momentum, name="Momentum"
                )

        loss_win = EpochWindow("training & validation loss", env=trainer.visdom_env)
        training_loss = trainer._loss_metric.get_loss(epoch)
        get_logger().info("epoch: %s, training loss: %s", epoch, training_loss)
        loss_win.plot_loss(epoch, training_loss, "training loss")

        inferencer = trainer.get_inferencer(phase=MachineLearningPhase.Validation)
        inferencer.inference()
        validation_loss = inferencer.loss.data.item()
        validation_acc = inferencer.accuracy_metric.get_accuracy(1)

        get_logger().info(
            "epoch: %s, learning_rate: %s, validation loss: %s, accuracy = %s",
            epoch,
            learning_rates,
            validation_loss,
            validation_acc,
        )
        loss_win = EpochWindow("training & validation loss", env=trainer.visdom_env)
        loss_win.plot_loss(epoch, validation_loss, "validation loss")
        EpochWindow("validation accuracy", env=trainer.visdom_env).plot_accuracy(
            epoch,
            validation_acc,
        )

        plot_class_accuracy = trainer.get_data("plot_class_accuracy")
        if plot_class_accuracy:
            class_accuracy = inferencer.accuracy_metric.get_class_accuracy(1)
            for idx, sub_list in enumerate(
                split_list_to_chunks(list(class_accuracy.keys()), 2)
            ):
                class_accuracy_win = EpochWindow(
                    "class accuracy part " + str(idx), env=trainer.visdom_env
                )
                for k in sub_list:
                    get_logger().info(
                        "epoch: %s, learning_rate: %s, class %s accuracy = %s",
                        epoch,
                        learning_rates,
                        k,
                        class_accuracy[k],
                    )
                    class_accuracy_win.plot_accuracy(
                        epoch,
                        class_accuracy[k],
                        "class_" + str(k) + "_accuracy",
                    )

        test_epoch_interval = 2
        if epoch % test_epoch_interval == 0 or epoch == trainer.hyper_parameter.epoch:
            inferencer = trainer.get_inferencer(phase=MachineLearningPhase.Test)
            inferencer.inference(per_class_accuracy=False)
            test_loss = inferencer.loss.data.item()
            test_acc = inferencer.accuracy_metric.get_accuracy(1)
            EpochWindow("test accuracy", env=trainer.visdom_env).plot_accuracy(
                epoch, test_acc, "accuracy"
            )
            get_logger().info(
                "epoch: %s, learning_rate: %s, test loss: %s, accuracy = %s",
                epoch,
                learning_rates,
                test_loss,
                test_acc,
            )
        Window.save_envs()
