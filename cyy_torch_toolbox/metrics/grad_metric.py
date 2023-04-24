import torch

from .metric import Metric


class GradMetric(Metric):
    __prev_parameter_list = None

    def _before_epoch(self, executor, **kwargs):
        if self.__prev_parameter_list is None:
            self.__prev_parameter_list = executor.model_util.get_parameter_list()

    def _after_epoch(self, epoch, executor, **kwargs):
        parameter_list = executor.model_util.get_parameter_list()
        self._set_epoch_metric(
            epoch,
            "grad_norm",
            torch.linalg.vector_norm(
                parameter_list - self.__prev_parameter_list.to(executor.device)
            ).item(),
        )
        self.__prev_parameter_list = parameter_list
