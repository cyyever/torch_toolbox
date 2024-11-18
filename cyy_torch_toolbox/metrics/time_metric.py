import time
from typing import Any

from .metric import Metric


class TimeMetric(Metric):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.__epoch_time_point: float = time.time()

    def _before_epoch(self, **kwargs: Any) -> None:
        self.__epoch_time_point = time.time()

    def _after_epoch(self, epoch: int, **kwargs: Any) -> None:
        self._set_epoch_metric(epoch, "duration", time.time() - self.__epoch_time_point)
