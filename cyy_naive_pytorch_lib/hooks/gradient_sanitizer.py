from callback import Callback


class GradientSanitizer(Callback):
    def _before_batch(self, **kwargs):
        # check parameters can be updated
        trainer = kwargs["model_executor"]
        optimizer = trainer.get_optimizer()
        for group in optimizer.param_groups:
            if group["params"] is trainer.model.parameters():
                return
        raise RuntimeError("can't find parameters in the optimizer")
