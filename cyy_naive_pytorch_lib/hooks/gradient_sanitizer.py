from callback import Callback


class GradientSanitizer(Callback):
    def _before_batch(self, **kwargs):
        batch_index = kwargs["batch_index"]
        if batch_index % 100 != 0:
            return
        # check parameters can be updated
        trainer = kwargs["model_executor"]
        optimizer = trainer.get_optimizer()
        for parameter in trainer.model.parameters():
            for group in optimizer.param_groups:
                if group["params"][0] is parameter:
                    return
            raise RuntimeError("can't find parameters in the optimizer")
