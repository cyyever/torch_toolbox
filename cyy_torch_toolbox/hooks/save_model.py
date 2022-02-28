import os
import shutil

from cyy_torch_toolbox.device import get_cpu_device
from cyy_torch_toolbox.hook import Hook
from ml_type import MachineLearningPhase


class SaveModelHook(Hook):
    __best_model = None
    __last_model = None
    save_epoch = False

    def _before_execute(self, **kwargs):
        self.__best_model = None
        self.__last_model = None

    def _after_epoch(self, **kwargs):
        trainer = kwargs["model_executor"]
        epoch = kwargs["epoch"]
        if self.save_epoch:
            model_dir = os.path.join(trainer.save_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "epoch_" + str(epoch) + ".pt")
            trainer.save_model(model_path)

        acc = trainer.get_inferencer_performance_metric(
            MachineLearningPhase.Validation
        ).get_epoch_metric(epoch, "accuracy")
        if not self.__best_model or acc > self.__best_model[1]:
            self.__best_model = (
                trainer.copy_model_with_loss().model.to(get_cpu_device()),
                acc,
            )

    def _after_execute(self, **kwargs):
        trainer = kwargs["model_executor"]
        model_dir = os.path.join(trainer.save_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        trainer.save_model(os.path.join(model_dir, "last.pt"))
        shutil.copy(
            self.__best_model[0],
            os.path.join(model_dir, "best_acc.pt"),
        )
