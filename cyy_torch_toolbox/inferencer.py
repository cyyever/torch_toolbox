import torch
import torch.nn as nn
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.ml_type import (MachineLearningPhase,
                                       ModelExecutorHookPoint,
                                       StopExecutingException)
from cyy_torch_toolbox.model_executor import ModelExecutor


class Inferencer(ModelExecutor):
    def inference(self, use_grad=False, epoch=None, **kwargs):
        self._prepare_execution(**kwargs)
        if epoch is None:
            epoch = 1
        self.exec_hooks(ModelExecutorHookPoint.BEFORE_EXECUTE)
        phase = self.phase
        if use_grad and self._model_with_loss.model_util.have_module(
            module_type=nn.RNNBase
        ):
            phase = MachineLearningPhase.Training
        with (torch.set_grad_enabled(use_grad), torch.cuda.stream(self.cuda_stream)):
            try:
                self.model.zero_grad(set_to_none=True)
                self.exec_hooks(
                    ModelExecutorHookPoint.BEFORE_EPOCH,
                    epoch=epoch,
                )
                self.exec_hooks(
                    ModelExecutorHookPoint.BEFORE_FETCH_BATCH, batch_index=0
                )
                for batch_index, batch in enumerate(self.dataloader):
                    self.exec_hooks(
                        ModelExecutorHookPoint.AFTER_FETCH_BATCH,
                        batch_index=batch_index,
                    )
                    batch_size, inputs, targets, other_info = self.decode_batch(batch)
                    if batch_size is None:
                        batch_size = self.get_batch_size(targets)
                    batch = (inputs, targets, other_info)

                    self.exec_hooks(
                        ModelExecutorHookPoint.BEFORE_BATCH,
                        batch_index=batch_index,
                        batch=batch,
                        batch_size=batch_size,
                    )

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
                        batch_index=batch_index,
                        inputs=result["inputs"],
                        input_features=result["input_features"],
                        targets=result["targets"],
                        batch_info=other_info,
                        batch_size=batch_size,
                        result=result,
                        epoch=epoch,
                    )
                    self.exec_hooks(
                        ModelExecutorHookPoint.BEFORE_FETCH_BATCH,
                        batch_index=batch_index + 1,
                    )
                self.exec_hooks(
                    ModelExecutorHookPoint.AFTER_EPOCH,
                    epoch=epoch,
                )
            except StopExecutingException:
                get_logger().warning("stop inference")
            finally:
                self._wait_stream()
            self.exec_hooks(ModelExecutorHookPoint.AFTER_EXECUTE)

    def get_gradient(self):
        self.inference(use_grad=True)
        return self.model_util.get_gradient_list()
