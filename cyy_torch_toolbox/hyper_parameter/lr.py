from typing import Any

import torch
from cyy_naive_lib.log import log_warning


def lr_scheduler_step_after_batch(
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> bool:
    return isinstance(
        lr_scheduler,
        torch.optim.lr_scheduler.OneCycleLR | torch.optim.lr_scheduler.CyclicLR,
    )
