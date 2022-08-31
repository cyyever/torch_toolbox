from cyy_torch_toolbox.hook import Hook


class LearningRateHook(Hook):
    def _before_batch(self, model_executor, **kwargs):
        model_executor.set_data(
            "cur_learning_rates",
            [group["lr"] for group in model_executor.get_optimizer().param_groups],
        )
