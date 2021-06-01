import torch
from cyy_naive_lib.log import get_logger

from dataset import DatasetUtil, decode_batch
from dataset_collection import DatasetCollection
from hyper_parameter import HyperParameter
from ml_type import MachineLearningPhase
from model_executor import ModelExecutor, ModelExecutorHookPoint
from model_util import ModelUtil
from model_with_loss import ModelWithLoss


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
        use_grad = kwargs.get("use_grad", False)
        epoch = kwargs.get("epoch", 1)
        self.exec_hooks(
            ModelExecutorHookPoint.BEFORE_EXECUTE,
            model_executor=self,
        )
        with torch.set_grad_enabled(use_grad):
            get_logger().debug("use device %s", self.device)
            self.model.zero_grad()
            self.exec_hooks(
                ModelExecutorHookPoint.BEFORE_EPOCH,
                model_executor=self,
                epoch=epoch,
            )
            for batch_index, batch in enumerate(self.dataloader):
                inputs, targets, _ = decode_batch(batch)
                result = self._model_with_loss(
                    inputs, targets, phase=self.phase, device=self.device
                )
                batch_loss = result["loss"].data.item()
                if use_grad:
                    real_batch_loss = result["loss"]
                    if self._model_with_loss.is_averaged_loss():
                        real_batch_loss *= self.get_batch_size(targets)
                    real_batch_loss /= len(self.dataset)
                    real_batch_loss.backward()

                self.exec_hooks(
                    ModelExecutorHookPoint.AFTER_BATCH,
                    model_executor=self,
                    batch=batch,
                    batch_loss=batch_loss,
                    batch_index=batch_index,
                    batch_size=self.get_batch_size(targets),
                    result=result,
                    epoch=epoch,
                )
            self.exec_hooks(
                ModelExecutorHookPoint.AFTER_EPOCH,
                model_executor=self,
                epoch=epoch,
            )
            return

    def get_gradient(self):
        self.inference(use_grad=True)
        return ModelUtil(self.model).get_gradient_list()
