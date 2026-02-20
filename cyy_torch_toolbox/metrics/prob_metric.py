from typing import Any

import torch
import torch.nn

from .classification_metric import ClassificationMetric


class ProbabilityMetric(ClassificationMetric):
    def get_prob(self, epoch: int) -> Any:
        return self.get_epoch_metric(epoch, "prob")

    def _after_batch(
        self, executor: Any, epoch: int, result: dict[str, Any], sample_indices: Any, **kwargs: Any
    ) -> None:
        output, _ = self._get_output(executor, result)
        last_layer = executor.model_util.get_last_underlying_module()
        epoch_prob = self.get_prob(epoch)
        if epoch_prob is None:
            epoch_prob = {}
        for sample_output, sample_index_tensor in zip(
            output, sample_indices, strict=False
        ):
            sample_index: int = int(sample_index_tensor.item())
            probs: None | torch.Tensor = None
            if isinstance(last_layer, torch.nn.LogSoftmax):
                probs = torch.exp(sample_output)
            elif isinstance(last_layer, torch.nn.Linear):
                probs = torch.nn.Softmax(dim=0)(sample_output)
            else:
                raise RuntimeError("unsupported layer", type(last_layer))
            assert probs is not None
            max_prob_index = int(torch.argmax(probs).item())
            epoch_prob[sample_index] = (
                max_prob_index,
                probs[max_prob_index].item(),
            )
        self._set_epoch_metric(epoch, "prob", epoch_prob)
