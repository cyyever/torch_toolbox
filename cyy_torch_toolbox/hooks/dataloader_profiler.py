import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter
from hook import Hook


class DataloaderProfiler(Hook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dataloader_time_counter = TimeCounter()

    def _before_fetch_batch(self, **kwargs):
        self.__dataloader_time_counter.reset_start_time()

    def _after_fetch_batch(self, **kwargs):
        get_logger().warning(
            "fetching batch used %sms",
            self.__dataloader_time_counter.elapsed_milliseconds(),
        )
