import torch

from .metric import Metric


class AccuracyMetric(Metric):
    __correct_count = None
    __dataset_size = None

    def get_accuracy(self, epoch):
        return self.get_epoch_metric(epoch, "accuracy").cpu()

    def _before_epoch(self, **kwargs):
        self.__dataset_size = 0
        self.__correct_count = None

    def _after_batch(self, **kwargs):
        output = kwargs["result"]["classification_output"]
        targets = kwargs["result"]["targets"]
        if len(output.shape) == 1:
            correct_count = (
                torch.eq(torch.round(output.sigmoid()), targets).view(-1).sum()
            )
        else:
            correct_count = (
                torch.eq(torch.max(output, dim=1)[1], targets).view(-1).sum()
            )
        if self.__correct_count is None:
            self.__correct_count = correct_count
        else:
            self.__correct_count += correct_count
        self.__dataset_size += targets.shape[0]

    def _after_epoch(self, **kwargs):
        epoch = kwargs["epoch"]
        accuracy = self.__correct_count / self.__dataset_size
        self._set_epoch_metric(epoch, "accuracy", accuracy)
