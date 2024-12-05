import os

from cyy_naive_lib.storage import DataStorage

from ..ml_type import MachineLearningPhase
from ..tensor import tensor_clone, tensor_to
from . import Hook


class KeepModelHook(Hook):
    __best_model: DataStorage = DataStorage(data=None)
    keep_best_model: bool = False
    save_epoch_model: bool = False
    save_last_model: bool = False
    save_best_model: bool = False

    @property
    def best_model(self) -> dict:
        return self.__best_model.data

    def __get_model_dir(self, root_dir: str) -> str:
        model_dir = os.path.join(root_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _before_execute(self, **kwargs) -> None:
        self.clear()

    def clear(self) -> None:
        self.__best_model.clear()

    def _after_validation(self, executor, epoch: int, **kwargs) -> None:
        trainer = executor
        if self.save_epoch_model:
            model_path = os.path.join(
                self.__get_model_dir(trainer.save_dir), f"epoch_{epoch}.pt"
            )
            trainer.save_model(model_path)

        if not self.save_best_model and not self.keep_best_model:
            return
        metric = trainer.get_cached_inferencer(
            MachineLearningPhase.Validation
        ).performance_metric.get_epoch_metrics(1)
        if (
            self.best_model is not None
            and metric["accuracy"]
            <= self.best_model["performance_metric"][MachineLearningPhase.Validation][
                "accuracy"
            ]
        ):
            return
        self.__best_model.set_data(
            {
                "epoch": epoch,
                "parameter": tensor_to(
                    data=trainer.model_util.get_parameters(detach=True),
                    non_blocking=True,
                    device="cpu",
                ),
                "performance_metric": {
                    MachineLearningPhase.Training: trainer.performance_metric.get_epoch_metrics(
                        epoch
                    ),
                    MachineLearningPhase.Validation: metric,
                },
            }
        )
        if self.save_best_model:
            assert trainer.save_dir is not None
            self.__best_model.set_data_path(
                os.path.join(self.__get_model_dir(trainer.save_dir), "best_model.pk")
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
