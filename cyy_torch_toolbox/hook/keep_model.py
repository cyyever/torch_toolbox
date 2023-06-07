import copy
import os

from cyy_naive_lib.storage import DataStorage
from cyy_torch_toolbox.device import get_cpu_device
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.tensor import tensor_to


class KeepModelHook(Hook):
    __best_model: DataStorage = DataStorage(data=None)
    keep_best_model: bool = False
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

    def _before_execute(self, **kwargs) -> None:
        self.clear()

    def clear(self) -> None:
        self.__best_model = DataStorage(data=None)

    def _after_validation(self, executor, epoch, **kwargs) -> None:
        trainer = executor
        if self.save_epoch_model:
            model_path = os.path.join(
                self.__get_model_dir(trainer.save_dir), f"epoch_{epoch}.pt"
            )
            trainer.save_model(model_path)

        if self.save_best_model or self.keep_best_model:
            acc = trainer.get_cached_inferencer(
                MachineLearningPhase.Validation
            ).performance_metric.get_epoch_metric(1, "accuracy")
            if self.best_model is None or acc > self.best_model[1]:
                self.__best_model.set_data(
                    (
                        tensor_to(
                            data=copy.deepcopy(trainer.model_util.get_parameter_dict()),
                            non_blocking=True,
                            device=get_cpu_device(),
                        ),
                        acc,
                        epoch,
                    )
                )
                if self.save_best_model:
                    assert trainer.save_dir is not None
                    self.__best_model.set_data_path(
                        os.path.join(
                            self.__get_model_dir(trainer.save_dir), "best_model.pk"
                        )
                    )
                    self.__best_model.save()

    def _after_execute(self, executor, **kwargs) -> None:
        trainer = executor
        if self.save_last_model:
            trainer.save_model(
                os.path.join(self.__get_model_dir(trainer.save_dir), "last.pt")
            )
        if self.save_best_model:
            self.__best_model.save()
