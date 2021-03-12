import torch

from dataset import DatasetUtil

from .metric import Metric


class AccuracyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.__classification_count_per_label: dict = dict()
        self.__classification_correct_count_per_label: dict = dict()
        self.__labels = None

    def get_accuracy(self, epoch):
        return self.get_epoch_metric(epoch, "accuracy")

    def get_class_accuracy(self, epoch):
        return self.get_epoch_metric(epoch, "class_accuracy")

    def _before_epoch(self, *args, **kwargs):
        model_exetutor = kwargs["model_exetutor"]
        self.__labels = DatasetUtil(model_exetutor.dataset).get_labels()
        for label in self.__labels:
            self.__classification_correct_count_per_label[label] = 0
            self.__classification_count_per_label[label] = 0

    def _after_batch(self, *args, **kwargs):
        batch = kwargs["batch"]
        targets = batch[1]
        result = kwargs["result"]
        output = result["output"]
        for target in targets:
            label = DatasetUtil.get_label_from_target(target)
            self.__classification_count_per_label[label] += 1
        correct = torch.eq(torch.max(output, dim=1)[1].cpu(), targets).view(-1)

        for label in self.__classification_correct_count_per_label:
            self.__classification_correct_count_per_label[label] += torch.sum(
                correct[targets == label]
            ).item()

    def _after_epoch(self, model_exetutor, epoch):
        accuracy = sum(self.__classification_correct_count_per_label.values()) / sum(
            self.__classification_count_per_label.values()
        )

        class_accuracy = dict()
        for label in self.__classification_correct_count_per_label:
            class_accuracy[label] = (
                self.__classification_correct_count_per_label[label]
                / self.__classification_count_per_label[label]
            )

        self._set_epoch_metric(epoch, "accuracy", accuracy)
        self._set_epoch_metric(epoch, "class_accuracy", class_accuracy)
