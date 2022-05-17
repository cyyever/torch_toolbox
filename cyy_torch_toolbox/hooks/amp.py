import torch
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.model_with_loss import ModelWithLoss

try:
    import apex

    has_apex = 1
except BaseException:
    has_apex = 0


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


if has_apex:

    class ApexAMP(Hook):
        def _before_execute(self, model_executor, **kwargs):
            # model_executor._model_with_loss.set_mod
            model_executor.model_with_loss.to(model_executor.device)
            model, optimizer = apex.amp.initialize(
                model_executor.model_with_loss.model,
                model_executor.get_optimizer(),
                opt_level="O1",
            )
            model_executor.set_model_with_loss(
                model_executor.model_with_loss.replace_model(model)
            )
            model_executor.remove_optimizer()
            model_executor.set_data("optimizer", optimizer)

        def _model_backward(self, model_executor, loss, **kwargs):
            with apex.amp.scale_loss(
                loss, model_executor.get_optimizer()
            ) as scaled_loss:
                scaled_loss.backward()
