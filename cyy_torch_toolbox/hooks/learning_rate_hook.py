from cyy_torch_toolbox.hook import Hook


class LearningRateHook(Hook):
    def _before_batch(self, executor, **kwargs):
        executor._data["cur_learning_rates"] = [
            group["lr"] for group in executor.get_optimizer().param_groups
        ]
