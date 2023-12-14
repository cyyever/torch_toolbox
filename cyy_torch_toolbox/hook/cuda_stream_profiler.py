import torch
from cyy_torch_toolbox.hook import Hook


class CUDAStreamProfiler(Hook):
    def _before_execute(self, **kwargs) -> None:
        if executor.device.type.lower() == "cuda":
            torch.cuda.set_sync_debug_mode("warn")

    def _after_execute(self, **kwargs) -> None:
        torch.cuda.set_sync_debug_mode("default")
