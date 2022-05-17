import torch
from cyy_torch_toolbox.hook import Hook


class AMP(Hook):
    __ctx = torch.autocast(device_type="cuda")

    def _model_forward(self, model_executor, model_kwargs, **kwargs):
        device = model_kwargs.get("device", None)
        if device is not None and "cuda" in str(device).lower():
            device_type = "cuda"
        else:
            device_type = "cpu"
        if device_type != self.__ctx.device:
            self.__ctx = torch.autocast(device_type=device_type)
        with self.__ctx:
            result = model_executor._model_with_loss(**model_kwargs)
            model_executor.set_data("forward_result", result)
