from typing import Any

import torch

from .metric import Metric


class GradMetric(Metric):
    __prev_parameter_list: Any = None

    def _before_epoch(self, executor, **kwargs: Any) -> None:
        if self.__prev_parameter_list is None:
            self.__prev_parameter_list = executor.model_util.get_parameter_list()

    def _after_epoch(self, epoch, executor, **kwargs) -> None:
        parameter_list = executor.model_util.get_parameter_list()
        self._set_epoch_metric(
            epoch,
            "grad_norm",
            torch.norm(
                parameter_list - self.__prev_parameter_list.to(executor.device)
            ).item(),
        )
        self.__prev_parameter_list = parameter_list
