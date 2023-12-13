import torch
from cyy_naive_lib.log import get_logger

from ..hook import Hook


class AMP(Hook):
    def __init__(self) -> None:
        assert torch.cuda.is_available()
        super().__init__(stripable=True)
        self.__ctx: None | torch.autocast = None
        self.__scaler = None
        self.__last_loss = None

    def _before_model_forward(self, executor, **kwargs) -> None:
        assert self._enabled
        device: torch.device = kwargs["evaluation_kwargs"]["device"]
        if self.__ctx is None or device.type != self.__ctx.device:
            self.__ctx = torch.autocast(device_type=device.type)
            executor._data["forward_contexts"].append(self.__ctx)

    def _model_backward(self, loss, **kwargs) -> None:
        assert self._enabled
        assert self.__ctx is not None
        assert self.__last_loss is None
        if (
            self.__scaler is None
            and self.__ctx is not None
            and str(self.__ctx.device) == "cuda"
        ):
            self.__scaler = torch.cuda.amp.GradScaler()
        assert self.__scaler is not None
        self.__last_loss = loss

    def _optimizer_step(self, executor, optimizer, **kwargs) -> None:
        assert self._enabled
        assert self.__scaler is not None
        assert self.__last_loss is not None
        while True:
            optimizer.zero_grad(set_to_none=True)
            self.__scaler.scale(self.__last_loss).backward()
            self.__scaler.step(optimizer)
            # Updates the scale for next iteration.
            self.__scaler.update()
            if self.__scaler._get_growth_tracker() == 0:
                get_logger().warning(
                    "found inf in AMP, scale is %s", self.__scaler._scale
                )
                continue
            break
        self.__last_loss = None
