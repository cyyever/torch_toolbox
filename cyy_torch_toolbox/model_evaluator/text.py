from .base import ModelEvaluator


class TextModelEvaluator(ModelEvaluator):
    def split_batch_input(self, inputs, targets, input_features=None) -> tuple:
        batch_dim = 0
        if isinstance(inputs, torch.Tensor):
            if (
                batch_dim == 0
                and inputs.shape[0] != targets.shape[0]
                and inputs.shape[1] == targets.shape[0]
            ):
                batch_dim = 1
            if batch_dim != 0:
                inputs = inputs.permute(batch_dim, 0)
        if batch_dim != 0 and isinstance(input_features, torch.Tensor):
            input_features = input_features.permute(batch_dim, 0, 2)
        return inputs, batch_dim, input_features
