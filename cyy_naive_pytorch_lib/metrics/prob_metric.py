import torch
import torch.nn as nn

from ml_type import ModelExecutorCallbackPoint
from model_executor import ModelExecutor

from .metric import Metric


class ProbabilityMetric(Metric):
    def __init__(self, model_exetutor: ModelExecutor):
        super().__init__(model_exetutor=model_exetutor)
        self.__probs: dict = dict()
        self.__cur_probs: dict = dict()

        self.add_callback(ModelExecutorCallbackPoint.BEFORE_EPOCH, self.__reset_prob)
        self.add_callback(ModelExecutorCallbackPoint.AFTER_BATCH, self.__compute_prob)
        self.add_callback(ModelExecutorCallbackPoint.AFTER_EPOCH, self.__save_prob)

    def get_prob(self, epoch):
        return self.__probs.get(epoch)

    def clear(self):
        self.__probs.clear()

    def __reset_prob(self, *args, **kwargs):
        self.__cur_probs.clear()

    def __compute_prob(self, *args, **kwargs):
        batch = kwargs["batch"]
        result = kwargs["result"]
        output = result["output"]
        last_layer = list(self._model_executor.model.modules())[-1]
        for i, sample_index in enumerate(batch[2]):
            sample_index = sample_index.data.item()

            probs = None
            if isinstance(last_layer, nn.LogSoftmax):
                probs = torch.exp(output[i])
            elif isinstance(last_layer, nn.Linear):
                probs = nn.Softmax()(output[i])
            else:
                raise RuntimeError("unsupported layer", type(last_layer))
            max_prob_index = torch.argmax(probs).data.item()
            self.__cur_probs[sample_index] = probs[max_prob_index].data.item()

    def __save_prob(self, model_exetutor,epoch):
        self.__probs[epoch] = self.__cur_probs
