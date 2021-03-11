import torch

from dataset import DatasetUtil
from ml_type import ModelExecutorCallbackPoint
from model_executor import ModelExecutor

from .metric import Metric


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
