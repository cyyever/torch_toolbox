from typing import Callable

import torch

from dataset import DatasetUtil
from ml_type import ModelExecutorCallbackPoint
from model_executor import ModelExecutor
from tensor import get_batch_size


class Metric:
    def __init__(self, model_exetutor: ModelExecutor):
        self._model_executor = model_exetutor
        self.__callback_names: list = []

    def add_callback(self, cb_point: ModelExecutorCallbackPoint, cb: Callable):
        name = self.__class__.__name__ + "." + str(cb)
        self._model_executor.add_named_callback(cb_point, name, cb)
        self.__callback_names.append(name)

    def remove_callbacks(self):
        for name in self.__callback_names:
            self._model_executor.remove_callback(name)


class LossMetric(Metric):
    def __init__(self, model_exetutor: ModelExecutor):
        super().__init__(model_exetutor=model_exetutor)
        self.__losses: dict = dict()
        self.__cur_epoch_loss = None
        self.add_callback(
            ModelExecutorCallbackPoint.BEFORE_EPOCH,
            self.__reset_epoch_loss,
        )
        self.add_callback(
            ModelExecutorCallbackPoint.AFTER_BATCH,
            self.__compute_batch_loss,
        )
        self.add_callback(
            ModelExecutorCallbackPoint.AFTER_EPOCH,
            self.__save_loss,
        )

    def clear(self):
        self.__losses.clear()

    def __reset_epoch_loss(self, *args, **kwargs):
        self.__cur_epoch_loss = None

    def __compute_batch_loss(self, *args, **kwargs):
        batch_loss = kwargs.get("batch_loss")
        batch = kwargs.get("batch")
        real_batch_loss = batch_loss
        if self._model_executor.model_with_loss.is_averaged_loss():
            real_batch_loss *= get_batch_size(
                self._model_executor.decode_batch(batch)[0]
            )
        real_batch_loss /= self._model_executor.get_data("dataset_size")
        if self.__cur_epoch_loss is None:
            self.__cur_epoch_loss = real_batch_loss
        else:
            self.__cur_epoch_loss += real_batch_loss

    def __save_loss(self, _, epoch):
        self.__losses[epoch] = self.__cur_epoch_loss

    @property
    def losses(self):
        assert self.__losses
        return self.__losses

    def get_loss(self, epoch):
        return self.__losses.get(epoch)

    @property
    def loss(self):
        assert len(self.losses) == 1
        for v in self.losses.values():
            return v
        return None


class AccuracyMetric(Metric):
    def __init__(self, model_exetutor: ModelExecutor):
        super().__init__(model_exetutor=model_exetutor)
        self.__labels = DatasetUtil(self._model_executor.dataset).get_labels()
        self.__classification_count_per_label: dict = dict()
        self.__classification_correct_count_per_label: dict = dict()
        self.__accuracies: dict = dict()
        self.__class_accuracies: dict = dict()

        self.add_callback(
            ModelExecutorCallbackPoint.BEFORE_EPOCH, self.__reset_epoch_count
        )
        self.add_callback(ModelExecutorCallbackPoint.AFTER_BATCH, self.__compute_acc)
        self.add_callback(ModelExecutorCallbackPoint.AFTER_EPOCH, self.__compute_acc)

    def get_accuracy(self, epoch):
        return self.__accuracies.get(epoch)

    def get_class_accuracy(self, epoch):
        return self.__class_accuracies.get(epoch)

    def clear(self):
        self.__accuracies.clear()
        self.__class_accuracies.clear()

    def __reset_epoch_count(self, *args, **kwargs):
        for label in self.__labels:
            self.__classification_correct_count_per_label[label] = 0
            self.__classification_count_per_label[label] = 0

    def __compute_count(self, *args, **kwargs):
        batch = kwargs["batch"]
        targets = batch[1]
        result = kwargs["result"]
        output = result["output"]
        for target in targets:
            label = DatasetUtil.get_label_from_target(target)
            self.__classification_count_per_label[label] += 1
        correct = torch.eq(torch.max(output, dim=1)[1], targets).view(-1)

        for label, count in self.__classification_correct_count_per_label.items():
            count += torch.sum(correct[targets == label]).item()

    def __compute_acc(self, epoch):
        accuracy = sum(self.__classification_correct_count_per_label.values()) / sum(
            self.__classification_count_per_label.values()
        )

        class_accuracy = dict()
        for label in self.__classification_correct_count_per_label:
            class_accuracy[label] = (
                self.__classification_correct_count_per_label[label]
                / self.__classification_count_per_label[label]
            )

        self.__accuracies[epoch] = accuracy
        self.__class_accuracies[epoch] = class_accuracy
