import torch

from .metric import Metric


class ClassAccuracyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.__classification_count_per_label: dict = {}
        self.__classification_correct_count_per_label: dict = {}
        self.__labels = None

    def get_class_accuracy(self, epoch):
        return self.get_epoch_metric(epoch, "class_accuracy")

    def _before_epoch(self, **kwargs):
        if not self.__labels:
            model_executor = kwargs["model_executor"]
            self.__labels = model_executor.dataset_collection.get_labels()
        for label in self.__labels:
            self.__classification_correct_count_per_label[label] = 0
            self.__classification_count_per_label[label] = 0

    def _after_batch(self, **kwargs):
        output = kwargs["result"]["classification_output"]
        targets = kwargs["result"]["targets"]
        assert isinstance(targets, torch.Tensor)
        correct = torch.eq(torch.max(output, dim=1)[1].cpu(), targets.cpu()).view(-1)
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

    def _after_epoch(self, **kwargs):
        epoch = kwargs["epoch"]

        class_accuracy = {}
        for label, cnt in self.__classification_correct_count_per_label.items():
            # non iid case
            if self.__classification_count_per_label[label] == 0:
                continue
            class_accuracy[label] = cnt / self.__classification_count_per_label[label]

        self._set_epoch_metric(epoch, "class_accuracy", class_accuracy)
