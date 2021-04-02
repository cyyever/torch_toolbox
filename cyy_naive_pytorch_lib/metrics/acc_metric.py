import torch
from cyy_naive_lib.log import get_logger
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
        model_executor = kwargs["model_executor"]
        self.__labels = DatasetUtil(model_executor.dataset).get_labels()
        for label in self.__labels:
            self.__classification_correct_count_per_label[label] = 0
            self.__classification_count_per_label[label] = 0

    def _after_batch(self, *args, **kwargs):
        batch = kwargs["batch"]
        targets = batch[1]
        result = kwargs["result"]
        output = result["output"]
        assert isinstance(targets, torch.Tensor)
        correct = torch.eq(torch.max(output, dim=1)[1].cpu(), targets).view(-1)
        assert correct.shape[0] == targets.shape[0]

        for label in self.__classification_correct_count_per_label:
            self.__classification_count_per_label[label] += torch.sum(
                targets == label
            ).item()
            self.__classification_correct_count_per_label[label] += torch.sum(
                correct[targets == label]
            ).item()
            assert (
                self.__classification_correct_count_per_label[label]
                <= self.__classification_count_per_label[label]
            )

    def _after_epoch(self, *args, **kwargs):
        epoch = kwargs["epoch"]
        accuracy = sum(self.__classification_correct_count_per_label.values()) / sum(
            self.__classification_count_per_label.values()
        )
        get_logger().info(
            "label count %s %s",
            sum(self.__classification_correct_count_per_label.values()),
            sum(self.__classification_count_per_label.values()),
        )

        class_accuracy = dict()
        for label in self.__classification_correct_count_per_label:
            class_accuracy[label] = (
                self.__classification_correct_count_per_label[label]
                / self.__classification_count_per_label[label]
            )
        get_logger().info("class_accuracy is %s", class_accuracy)

        self._set_epoch_metric(epoch, "accuracy", accuracy)
        self._set_epoch_metric(epoch, "class_accuracy", class_accuracy)
