import os

from cyy_naive_lib.storage import DataStorage
from cyy_torch_toolbox.device import get_cpu_device
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class KeepModelHook(Hook):
    __best_model: DataStorage = DataStorage(data=None)
    save_epoch_model: bool = False
    save_last_model: bool = False
    save_best_model: bool = False

    @property
    def best_model(self):
        return self.__best_model.data

    def __get_model_dir(self, root_dir: str) -> str:
        model_dir = os.path.join(root_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def offload_from_memory(self):
        self.__best_model.save()

    def _before_execute(self, **kwargs):
        self.__best_model = DataStorage(data=None)

    def _after_validation(self, model_executor, epoch, **kwargs):
        trainer = model_executor
        if self.save_epoch_model:
            model_path = os.path.join(
                self.__get_model_dir(trainer.save_dir), f"epoch_{epoch}.pt"
            )
            trainer.save_model(model_path)

        acc = trainer.get_inferencer_performance_metric(
            MachineLearningPhase.Validation
        ).get_epoch_metric(epoch, "accuracy")
        if self.best_model is None or acc > self.best_model[1]:
            self.__best_model.set_data(
                (
                    trainer.copy_model_with_loss().model.to(
                        get_cpu_device(), non_blocking=True
                    ),
                    acc,
                )
            )
            if trainer.save_dir is not None:
                self.__best_model.set_data_path(
                    os.path.join(
                        self.__get_model_dir(trainer.save_dir), "best_model.pk"
                    )
                )
                self.__best_model.save()

    def _after_execute(self, model_executor, **kwargs):
        trainer = model_executor
        if self.save_last_model:
            trainer.save_model(
                os.path.join(self.__get_model_dir(trainer.save_dir), "last.pt")
            )
        if self.save_best_model:
            self.__best_model.set_data(self.best_model[0])
            self.__best_model.save()
        else:
            self.__best_model.clear()
