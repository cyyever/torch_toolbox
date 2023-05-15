from cyy_torch_toolbox.hook import Hook


class LearningRateHook(Hook):
    learning_rates = None

    def _before_batch(self, executor, **kwargs):
        self.learning_rates = [
            group["lr"] for group in executor.get_optimizer().param_groups
        ]
