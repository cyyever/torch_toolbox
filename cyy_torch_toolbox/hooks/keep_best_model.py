from cyy_naive_lib.log import get_logger
from device import get_cpu_device
from hook import Hook
from ml_type import MachineLearningPhase


class KeepBestModelHook(Hook):
    __best_epoch = None
    __best_model = None

    def _before_execute(self, **kwargs):
        self.__best_epoch = None
        self.__best_model = None

    def _after_epoch(self, **kwargs):
        trainer = kwargs["model_executor"]
        epoch = kwargs["epoch"]
        acc = trainer.get_inferencer_performance_metric(
            MachineLearningPhase.Validation
        ).get_epoch_metric(epoch, "accuracy")
        if not self.__best_epoch or acc > self.__best_epoch[1]:
            self.__best_epoch = (epoch, acc)
            self.__best_model = trainer.copy_model_with_loss().model.to(get_cpu_device())

    @property
    def best_model(self):
        assert self.__best_model is not None
        get_logger().info("get best model from epoch %s", self.__best_epoch[0])
        return self.__best_model
