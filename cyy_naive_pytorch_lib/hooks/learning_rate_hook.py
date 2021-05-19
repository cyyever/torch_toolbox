from hook import Callback


class LearningRateHook(Callback):
    def _before_batch(self, **kwargs):
        model_executor = kwargs["model_executor"]
        model_executor.set_data(
            "cur_learning_rates",
            [group["lr"] for group in model_executor.get_optimizer().param_groups],
        )
