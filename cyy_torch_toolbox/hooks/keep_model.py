import os
import shutil

from cyy_torch_toolbox.device import get_cpu_device
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class KeepModelHook(Hook):
    __best_model: tuple = None
    save_flag: bool = False

    @property
    def best_model(self):
        return self.__best_model

    def _before_execute(self, **kwargs):
        self.__best_model = None

    def _after_epoch(self, **kwargs):
        trainer = kwargs["model_executor"]
        epoch = kwargs["epoch"]
        if self.save_flag:
            model_dir = os.path.join(trainer.save_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "epoch_" + str(epoch) + ".pt")
            trainer.save_model(model_path)

        acc = trainer.get_inferencer_performance_metric(
            MachineLearningPhase.Validation
        ).get_epoch_metric(epoch, "accuracy")
        if self.__best_model is None or acc > self.__best_model[1]:
            self.__best_model = (
                trainer.copy_model_with_loss().model.to(get_cpu_device()),
                acc,
            )

    def _after_execute(self, **kwargs):
        if not self.save_flag:
            return
        trainer = kwargs["model_executor"]
        model_dir = os.path.join(trainer.save_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        trainer.save_model(os.path.join(model_dir, "last.pt"))
        shutil.copy(
            self.__best_model[0],
            os.path.join(model_dir, "best_acc.pt"),
        )
