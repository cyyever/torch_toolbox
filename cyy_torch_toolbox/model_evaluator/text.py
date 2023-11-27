import torch

from .base import ModelEvaluator


class TextModelEvaluator(ModelEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs.get("tokenizer", None)
        word_vector_name = kwargs.get("word_vector_name", None)
        if word_vector_name is not None:
            from .word_vector import PretrainedWordVector

            PretrainedWordVector(word_vector_name).load_to_model(
                model_evaluator=model_evaluator,
                tokenizer=self.tokenizer,
                freeze_embedding=model_kwargs.get("freeze_word_vector", False),
            )

    def get_feature_forward_fun(self) -> str:
        return "forward_input_feature"

    def split_batch_input(self, inputs, targets) -> dict:
        batch_dim: int = 0
        if isinstance(inputs, torch.Tensor):
            if (
                batch_dim == 0
                and inputs.shape[0] != targets.shape[0]
                and inputs.shape[1] == targets.shape[0]
            ):
                batch_dim = 1
            if batch_dim != 0:
                inputs = inputs.permute(batch_dim, 0)
        return {"inputs": inputs, "batch_dim": batch_dim}
