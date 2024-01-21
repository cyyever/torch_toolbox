import torch
import torch.nn

from .metric import Metric


class ProbabilityMetric(Metric):
    def get_prob(self, epoch: int):
        return self.get_epoch_metric(epoch, "prob")

    def _after_batch(self, executor, epoch, result, sample_indices, **kwargs) -> None:
        output = result["classification_output"]
        last_layer = list(executor.model.modules())[-1]
        epoch_prob = self.get_prob(epoch)
        if epoch_prob is None:
            epoch_prob = {}
        for i, sample_index in enumerate(sample_indices):
            sample_index: int = sample_index.item()
            probs: None | torch.Tensor = None
            if isinstance(last_layer, torch.nn.LogSoftmax):
                probs = torch.exp(output[i])
            elif isinstance(last_layer, torch.nn.Linear):
                probs = torch.nn.Softmax(dim=0)(output[i])
            else:
                raise RuntimeError("unsupported layer", type(last_layer))
            assert probs is not None
            max_prob_index = torch.argmax(probs).data.item()
            epoch_prob[sample_index] = (
                max_prob_index,
                probs[max_prob_index].item(),
            )
        self._set_epoch_metric(epoch, "prob", epoch_prob)
