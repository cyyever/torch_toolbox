import torch
from cyy_torch_toolbox.hook import Hook


class CUDAStreamProfiler(Hook):
    def _before_execute(self, **kwargs) -> None:
        torch.cuda.set_sync_debug_mode(1)

    def _after_execute(self, **kwargs) -> None:
        torch.cuda.set_sync_debug_mode(0)
