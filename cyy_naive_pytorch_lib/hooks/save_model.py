import os
import shutil

import torch
from callback import Callback
from ml_type import MachineLearningPhase


class SaveModelHook(Callback):
    def __init__(self):
        super().__init__()
        self.__best_epoch = None
        self.__last_epoch = None
        self.__model_paths = None
        self.__model_dir = None

    def _before_execute(self, **kwargs):
        self.__best_epoch = None
        self.__model_paths = dict()
        self.__model_dir = None

    def _after_epoch(self, **kwargs):
        trainer = kwargs["model_executor"]
        epoch = kwargs["epoch"]
        self.__model_dir = os.path.join(trainer.save_dir, "model")
        os.makedirs(self.__model_dir, exist_ok=True)
        model_path = os.path.join(self.__model_dir, "epoch_" + str(epoch) + ".pt")
        torch.save(trainer.model, model_path)
        self.__model_paths[epoch] = model_path
        acc = trainer.get_validation_metric(
            MachineLearningPhase.Validation
        ).get_epoch_metric(epoch, "accuracy")
        if not self.__best_epoch or acc > self.__best_epoch[1]:
            self.__best_epoch = (epoch, acc)
        self.__last_epoch = epoch

    def _after_execute(self, **kwargs):
        shutil.copy(
            self.__model_paths[self.__last_epoch],
            os.path.join(self.__model_dir, "last.pt"),
        )
        shutil.copy(
            self.__model_paths[self.__best_epoch[0]],
            os.path.join(self.__model_dir, "best_acc.pt"),
        )
