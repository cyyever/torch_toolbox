from typing import Any

import torch
from cyy_naive_lib.log import log_debug

from . import Hook


class BackendBenchmarkHook(Hook):
    def _before_execute(self, **kwargs: Any) -> None:
        if torch.cuda.is_available() and not torch.are_deterministic_algorithms_enabled():
            torch.backends.cudnn.benchmark = True
            log_debug("benchmark cudnn")

    def _after_execute(self, **kwargs: Any) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
