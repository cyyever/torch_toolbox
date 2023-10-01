import torch
from torchmetrics import F1Score

from .metric import Metric


class F1Metric(Metric):
    __f1_score_counter: None | F1Score = None

    def _before_execution(self, **kwargs) -> None:
        self.__dataset_size = 0
        self.__correct_count = None

    def get_accuracy(self, epoch: int) -> float:
        acc = self.get_epoch_metric(epoch, "accuracy")
        if isinstance(acc, torch.Tensor):
            return acc.cpu().item()
        return acc

    def _before_epoch(self, **kwargs) -> None:
        executor=kwargs["executor"]
        self.__f1_score_counter=F1Score(num_classes=executor.dataset_collection.)

        self.__dataset_size = 0
        self.__correct_count = None

    def _after_batch(self, result, **kwargs) -> None:
        output = result["model_output"]
        logits = result.get("logits", None)
        targets = result["targets"]
        if logits is not None:
            output = logits
        correct_count = 0
        if output.shape == targets.shape:
            if len(targets.shape) == 2:
                for idx, maxidx in enumerate(torch.argmax(output, dim=1)):
                    if targets[idx][maxidx] == 1:
                        correct_count += 1
            elif len(targets.shape) <= 1:
                correct_count = (
                    torch.eq(torch.round(output.sigmoid()), targets).view(-1).sum()
                )
            else:
                raise NotImplementedError()
        else:
            correct_count = (
                torch.eq(torch.max(output, dim=1)[1], targets).view(-1).sum()
            )
        if self.__correct_count is None:
            self.__correct_count = correct_count
        else:
            self.__correct_count += correct_count
        self.__dataset_size += targets.shape[0]

    def _after_epoch(self, **kwargs) -> None:
        epoch = kwargs["epoch"]
        accuracy = self.__correct_count / self.__dataset_size
        self._set_epoch_metric(epoch, "accuracy", accuracy)
