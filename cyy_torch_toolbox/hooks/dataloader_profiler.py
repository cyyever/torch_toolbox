from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox.hook import Hook


class DataloaderProfiler(Hook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dataloader_time_counter = TimeCounter()

    def _before_fetch_batch(self, **kwargs):
        batch_index = kwargs["batch_index"]
        if batch_index == 0 or batch_index % 100 == 0:
            self.__dataloader_time_counter.reset_start_time()

    def _after_fetch_batch(self, **kwargs):
        batch_index = kwargs["batch_index"]
        if batch_index == 0 or batch_index % 100 == 0:
            get_logger().warning(
                "fetching batch %s used %sms",
                batch_index,
                self.__dataloader_time_counter.elapsed_milliseconds(),
            )
