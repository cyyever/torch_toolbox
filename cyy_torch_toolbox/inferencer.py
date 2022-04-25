import torch
import torch.nn as nn

from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.hyper_parameter import HyperParameter
from cyy_torch_toolbox.ml_type import (DatasetType, MachineLearningPhase,
                                       ModelExecutorHookPoint)
from cyy_torch_toolbox.model_executor import ModelExecutor
from cyy_torch_toolbox.model_with_loss import ModelWithLoss


class Inferencer(ModelExecutor):
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset_collection: DatasetCollection,
        phase: MachineLearningPhase,
        hyper_parameter: HyperParameter,
    ):
        super().__init__(model_with_loss, dataset_collection, phase, hyper_parameter)
        assert self.phase != MachineLearningPhase.Training

    def inference(self, **kwargs):
        self._prepare_execution()
        use_grad = kwargs.get("use_grad", False)
        epoch = kwargs.get("epoch", 1)
        self.exec_hooks(ModelExecutorHookPoint.BEFORE_EXECUTE)
        if use_grad and self.dataset_collection.dataset_type == DatasetType.Text:
            # cudnn rnn backward needs training model
            self._model_with_loss.set_model_mode(self.phase)
            self._model_with_loss.model_util.change_sub_modules(
                nn.RNNBase, lambda _, v, *__: v.train()
            )
        with torch.set_grad_enabled(use_grad):
            with torch.cuda.stream(self.cuda_stream):
                if use_grad:
                    self.model.zero_grad(set_to_none=True)
                self.exec_hooks(
                    ModelExecutorHookPoint.BEFORE_EPOCH,
                    epoch=epoch,
                )
                self.exec_hooks(ModelExecutorHookPoint.BEFORE_FETCH_BATCH)
                for batch_index, batch in enumerate(self.dataloader):
                    self.exec_hooks(ModelExecutorHookPoint.AFTER_FETCH_BATCH)
                    batch_size, inputs, targets, other_info = self.decode_batch(batch)
                    if batch_size is None:
                        batch_size = self.get_batch_size(targets)
                    batch = (inputs, targets, other_info)
                    result = self._model_with_loss(
                        inputs,
                        targets,
                        phase=self.phase,
                        device=self.device,
                        non_blocking=True,
                        batch_size=batch_size,
                    )
                    batch_loss = result["loss"]
                    if use_grad:
                        real_batch_loss = result["normalized_loss"] / len(self.dataset)
                        real_batch_loss.backward()

                    self.exec_hooks(
                        ModelExecutorHookPoint.AFTER_BATCH,
                        batch=batch,
                        batch_loss=batch_loss,
                        inputs=result["inputs"],
                        input_embeddings=result["input_embeddings"],
                        targets=result["targets"],
                        normalized_batch_loss=result["normalized_loss"],
                        batch_index=batch_index,
                        batch_size=self.get_batch_size(targets),
                        result=result,
                        epoch=epoch,
                    )
                    self.exec_hooks(ModelExecutorHookPoint.BEFORE_FETCH_BATCH)
                self.exec_hooks(
                    ModelExecutorHookPoint.AFTER_EPOCH,
                    epoch=epoch,
                )
        self.exec_hooks(ModelExecutorHookPoint.AFTER_EXECUTE)

    def get_gradient(self):
        self.inference(use_grad=True)
        return self.model_util.get_gradient_list()
