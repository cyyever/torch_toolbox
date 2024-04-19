import torch
from cyy_naive_lib.log import log_debug

from . import Hook


class CUDNNHook(Hook):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert torch.cuda.is_available()

    def _before_execute(self, **kwargs) -> None:
        if not torch.are_deterministic_algorithms_enabled():
            torch.backends.cudnn.benchmark = True
            log_debug("benchmark cudnn")

    def _after_execute(self, **kwargs) -> None:
        torch.backends.cudnn.benchmark = False
