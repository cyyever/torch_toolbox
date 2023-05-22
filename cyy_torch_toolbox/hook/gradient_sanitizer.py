from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class GradientSanitizer(Hook):
    def _before_batch(self, executor, batch_index, **kwargs):
        if executor.phase != MachineLearningPhase.Training:
            return
        if batch_index % 100 != 0:
            return
        # check parameters can be updated
        trainer = executor
        optimizer = trainer.get_optimizer()
        for name, parameter in trainer.model.named_parameters():
            flag = False
            for group in optimizer.param_groups:
                if flag:
                    break
                for param in group["params"]:
                    if param is parameter:
                        flag = True
                        break
            if not flag:
                raise RuntimeError("can't find parameter " + name + " in the optimizer")
