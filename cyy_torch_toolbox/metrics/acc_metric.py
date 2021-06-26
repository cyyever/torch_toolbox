import torch

from .metric import Metric


class AccuracyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.__correct_count = None
        self.__dataset_size = None
        self.__pending_count = None

    def get_accuracy(self, epoch):
        return self.get_epoch_metric(epoch, "accuracy").cpu()

    def _before_epoch(self, **kwargs):
        self.__dataset_size = 0
        self.__correct_count = 0
        self.__pending_count = None

    def _after_batch(self, **kwargs):
        batch = kwargs["batch"]
        targets = batch[1].detach()
        output = kwargs["result"]["output"].detach()
        correct_count = torch.eq(torch.max(output, dim=1)[1], targets).view(-1).sum()
        self.__process_pending_count()
        self.__pending_count = correct_count
        self.__dataset_size += targets.shape[0]

    def __process_pending_count(self):
        if self.__pending_count is not None:
            self.__correct_count += self.__pending_count.cpu()
            self.__pending_count = None

    def _after_epoch(self, **kwargs):
        epoch = kwargs["epoch"]
        self.__process_pending_count()
        accuracy = self.__correct_count / self.__dataset_size
        self._set_epoch_metric(epoch, "accuracy", accuracy)
