import torch
import torch.nn as nn

from cyy_torch_toolbox.ml_type import (MachineLearningPhase,
                                       ModelExecutorHookPoint)
from cyy_torch_toolbox.model_executor import ModelExecutor


class Inferencer(ModelExecutor):
    def inference(self, **kwargs):
        self._prepare_execution()
        use_grad = kwargs.get("use_grad", False)
        epoch = kwargs.get("epoch", 1)
        self.exec_hooks(ModelExecutorHookPoint.BEFORE_EXECUTE)
        phase = self.phase
        if use_grad and self._model_with_loss.model_util.have_module(
            module_type=nn.RNNBase
        ):
            phase = MachineLearningPhase.Training
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
                        phase=phase,
                        device=self.device,
                        non_blocking=True,
                    )
                    if use_grad:
                        real_batch_loss = result["loss"]
                        if result["is_averaged_loss"]:
                            real_batch_loss *= batch_size
                        real_batch_loss /= self.dataset_size
                        real_batch_loss.backward()

                    self.exec_hooks(
                        ModelExecutorHookPoint.AFTER_BATCH,
                        batch=batch,
                        inputs=result["inputs"],
                        input_features=result["input_features"],
                        targets=result["targets"],
                        batch_index=batch_index,
                        batch_size=batch_size,
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
