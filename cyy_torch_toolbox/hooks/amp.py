import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.hook import Hook


class AMP(Hook):
    __ctx = torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    __scaler = None

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

    def _model_backward(self, loss, **kwargs):
        if self.__scaler is None and str(self.__ctx.device) != "cpu":
            self.__scaler = torch.cuda.amp.GradScaler()
        if self.__scaler is not None:
            self.__scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self, model_executor, **kwargs):
        optimizer = model_executor.get_optimizer()
        if self.__scaler is None:
            optimizer.step()
            return
        self.__scaler.step(optimizer)
        if sum(
            self.__scaler._found_inf_per_device(optimizer=optimizer).values()
        ).item():
            model_executor.set_data("step_skipped", True)
            get_logger().debug("found inf in AMP")

        # Updates the scale for next iteration.
        self.__scaler.update()
