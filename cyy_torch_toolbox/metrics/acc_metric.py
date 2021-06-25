import torch
from dataset import decode_batch

from .metric import Metric


class AccuracyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.__classification_count_per_label: dict = dict()
        self.__classification_correct_count_per_label: dict = dict()
        self.__labels = None
        self.__correct_count = None
        self.__dataset_size = None

    def get_accuracy(self, epoch):
        return self.get_epoch_metric(epoch, "accuracy")

    def _before_epoch(self, **kwargs):
        if not self.__labels:
            model_executor = kwargs["model_executor"]
            self.__labels = model_executor.dataset_collection.get_labels()
        self.__dataset_size = 0
        self.__correct_count = 0

    def _after_batch(self, **kwargs):
        batch = kwargs["batch"]
        targets = decode_batch(batch)[1]
        result = kwargs["result"]
        output = result["output"]
        assert isinstance(targets, torch.Tensor)
        correct = torch.eq(torch.max(output, dim=1)[1].cpu(), targets.cpu()).view(-1)
        assert correct.shape[0] == targets.shape[0]
        self.__correct_count += torch.sum(correct)
        self.__dataset_size += targets.shape[0]

        assert self.__correct_count <= self.__dataset_size

    def _after_epoch(self, **kwargs):
        epoch = kwargs["epoch"]
        accuracy = sum(self.__classification_correct_count_per_label.values()) / sum(
            self.__classification_count_per_label.values()
        )
        self._set_epoch_metric(epoch, "accuracy", accuracy)
