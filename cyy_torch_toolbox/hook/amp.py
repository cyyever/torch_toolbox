import torch
from cyy_naive_lib.log import get_logger

from ..hook import Hook


class AMP(Hook):
    def __init__(self, *args, **kwargs) -> None:
        assert torch.cuda.is_available()
        super().__init__(*args, **kwargs)
        self.__ctx = None
        self.__scaler = None

    def _model_forward(self, executor, model_kwargs, **kwargs) -> None:
        assert self._enabled
        device = model_kwargs.get("device", None)
        if device is not None and "cuda" in str(device).lower():
            device_type = "cuda"
        else:
            device_type = "cpu"
        if self.__ctx is None or device_type != self.__ctx.device:
            self.__ctx = torch.autocast(device_type=device_type)
        with self.__ctx:
            result = executor.running_model_evaluator(**model_kwargs)
            executor._data["forward_result"] = result

    def _model_backward(self, loss, **kwargs) -> None:
        assert self._enabled
        if (
            self.__scaler is None
            and self.__ctx is not None
            and str(self.__ctx.device) != "cpu"
        ):
            self.__scaler = torch.cuda.amp.GradScaler()
        if self.__scaler is not None:
            self.__scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self, executor, **kwargs) -> None:
        assert self._enabled
        optimizer = executor.get_optimizer()
        if self.__scaler is None:
            optimizer.step()
            return
        self.__scaler.step(optimizer)
        if (
            hasattr(optimizer, "_step_supports_amp_scaling")
            and optimizer._step_supports_amp_scaling
        ):
            pass
        else:
            if sum(
                self.__scaler._found_inf_per_device(optimizer=optimizer).values()
            ).item():
                executor._data["step_skipped"] = True
                get_logger().warning(
                    "found inf in AMP, scale is %s", self.__scaler._scale
                )

        # Updates the scale for next iteration.
        self.__scaler.update()
