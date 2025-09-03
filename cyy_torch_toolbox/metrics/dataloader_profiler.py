from cyy_naive_lib.time_counter import TimeCounter

from .metric import Metric


class DataloaderProfiler(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__dataloader_time_counter = TimeCounter()
        self.__accumulated_time: float = 0

    def _before_epoch(self, **kwargs) -> None:
        self.__accumulated_time = 0

    def _before_fetch_batch(self, **kwargs) -> None:
        self.__dataloader_time_counter.reset_start_time()

    def _after_fetch_batch(self, **kwargs) -> None:
        self.__accumulated_time += self.__dataloader_time_counter.elapsed_milliseconds()

    def _after_epoch(self, executor, epoch, **kwargs) -> None:
        self._set_epoch_metric(
            epoch, "data_waiting_time", self.__accumulated_time / 1000
        )
