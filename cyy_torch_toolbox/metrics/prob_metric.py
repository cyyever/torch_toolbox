import torch
import torch.nn as nn

from hooks.add_index_to_dataset import AddIndexToDataset

from .metric import Metric


class ProbabilityMetric(Metric, AddIndexToDataset):
    def get_prob(self, epoch):
        return self.get_epoch_metric(epoch, "prob")

    def _before_execute(self, **kwargs):
        Metric._before_execute(self, **kwargs)
        AddIndexToDataset._before_execute(self, **kwargs)

    def _after_batch(self, **kwargs):
        batch = kwargs["batch"]
        epoch = kwargs["epoch"]
        model_executor = kwargs["model_executor"]
        result = kwargs["result"]
        output = result["output"]
        last_layer = list(model_executor.model.modules())[-1]
        epoch_prob = self.get_prob(epoch)
        if epoch_prob is None:
            epoch_prob = dict()
        for i, sample_index in enumerate(batch[2]["index"]):
            sample_index = sample_index.data.item()
            probs = None
            if isinstance(last_layer, nn.LogSoftmax):
                probs = torch.exp(output[i])
            elif isinstance(last_layer, nn.Linear):
                probs = nn.Softmax(dim=0)(output[i])
            else:
                raise RuntimeError("unsupported layer", type(last_layer))
            max_prob_index = torch.argmax(probs).data.item()
            epoch_prob[sample_index] = (
                max_prob_index,
                probs[max_prob_index].data.item(),
            )
        self._set_epoch_metric(epoch, "prob", epoch_prob)

    def _after_execute(self, **kwargs):
        AddIndexToDataset._after_execute(self, **kwargs)
