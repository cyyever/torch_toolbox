import torch
from cyy_torch_toolbox.hook import Hook


class ClassificationMetric(Hook):
    def _get_output(self, result: dict) -> torch.Tensor:
        output = result["model_output"]
        logits = result.get("logits", None)
        if logits is not None:
            output = logits
        if output.shape == 2:
            output = torch.max(output, dim=1).review(-1)
        if (output < 0).sum().item():
            output = output.sigmoid()
        return output
