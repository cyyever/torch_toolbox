import copy
from typing import Optional

import torch
import torch.nn as nn

from hyper_parameter import HyperParameter
from device import get_device, put_data_to_device
from model_loss import ModelWithLoss
from model_util import ModelUtil
from util import get_batch_size
from dataset import DatasetUtil, split_dataset_by_label
from local_types import MachineLearningPhase


class Inferencer:
    @staticmethod
    def prepend_callback(kwargs, name, new_fun):
        callbacks = kwargs.get(name, [])
        callbacks.insert(0, new_fun)
        kwargs[name] = callbacks
        return kwargs

    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset,
        phase: MachineLearningPhase,
        hyper_parameter: Optional[HyperParameter] = None,
    ):
        assert phase != MachineLearningPhase.Training
        self.__model_with_loss = copy.deepcopy(model_with_loss)
        self.__dataset = dataset
        self.__phase = phase
        self.__hyper_parameter = hyper_parameter

    @property
    def model_with_loss(self):
        return self.__model_with_loss

    @property
    def model(self):
        return self.model_with_loss.model

    @property
    def dataset(self):
        return self.__dataset

    def set_dataset(self, dataset):
        self.__dataset = dataset

    def load_model(self, model_path):
        self.model_with_loss.set_model(
            torch.load(model_path, map_location=get_device())
        )

    def inference(self, **kwargs):
        data_loader = self.__hyper_parameter.get_dataloader(
            self.__dataset, self.__phase
        )

        use_grad = kwargs.get("use_grad", False)
        with torch.set_grad_enabled(use_grad):
            device = get_device()
            self.model_with_loss.set_model_mode(self.__phase)
            self.model.zero_grad()
            self.model.to(device)
            total_loss = torch.zeros(1)
            total_loss = total_loss.to(device)
            for batch in data_loader:
                inputs = put_data_to_device(batch[0], device)
                targets = put_data_to_device(batch[1], device)
                real_batch_size = get_batch_size(inputs)

                result = self.model_with_loss(
                    inputs, targets, phase=self.__phase)
                batch_loss = result["loss"]

                for callback in kwargs.get("after_batch_callbacks", []):
                    callback(batch, result, targets)

                normalized_batch_loss = batch_loss
                if self.model_with_loss.is_averaged_loss():
                    normalized_batch_loss *= real_batch_size
                normalized_batch_loss /= len(self.__dataset)
                if use_grad:
                    normalized_batch_loss.backward()
                total_loss += normalized_batch_loss
            return total_loss

    def get_gradient(self):
        self.inference(use_grad=True)
        return ModelUtil(self.model).get_gradient_list()


class ClassificationInferencer(Inferencer):
    def inference(self, **kwargs):
        class_count = dict()
        class_correct_count = dict()
        instance_output = dict()
        instance_prob = dict()
        per_sample_prob = kwargs.get("per_sample_prob", False)

        dataset_util = DatasetUtil(self.dataset)
        tmp = split_dataset_by_label(self.dataset)
        for label in dataset_util.get_labels():
            class_correct_count[label] = 0
            class_count[label] = len(tmp[label]["indices"])

        def after_batch_callback(batch, result, targets):
            nonlocal per_sample_prob
            nonlocal instance_output
            output = result["output"]
            if per_sample_prob:
                for i, instance_index in enumerate(batch[2]):
                    instance_index = instance_index.data.item()
                    instance_output[instance_index] = output[i]
            correct = torch.eq(torch.max(output, dim=1)[1], targets).view(-1)

            for label in class_correct_count:
                class_correct_count[label] += torch.sum(
                    correct[targets == label]
                ).item()

        kwargs = Inferencer.prepend_callback(
            kwargs, "after_batch_callbacks", after_batch_callback
        )
        loss = super().inference(**kwargs)

        for label in class_count:
            class_count[label] = class_correct_count[label] / \
                class_count[label]
        if per_sample_prob:
            last_layer = list(self.model.modules())[-1]
            if isinstance(last_layer, nn.LogSoftmax):
                for k, v in instance_output.items():
                    probs = torch.exp(v)
                    max_prob_index = torch.argmax(probs).data.item()
                    instance_prob[k] = (
                        max_prob_index,
                        probs[max_prob_index].data.item(),
                    )
            elif isinstance(last_layer, nn.Linear):
                for k, v in instance_output.items():
                    prob_v = nn.Softmax()(v)
                    max_prob_index = torch.argmax(prob_v).data.item()
                    instance_prob[k] = (
                        max_prob_index,
                        prob_v[max_prob_index].data.item(),
                    )
            else:
                raise RuntimeError("unsupported layer", type(last_layer))

        for label in class_count:
            class_count[label] = class_correct_count[label] / \
                class_count[label]
        if per_sample_prob:
            last_layer = list(self.model.modules())[-1]
            if isinstance(last_layer, nn.LogSoftmax):
                for k, v in instance_output.items():
                    probs = torch.exp(v)
                    max_prob_index = torch.argmax(probs).data.item()
                    instance_prob[k] = (
                        max_prob_index,
                        probs[max_prob_index].data.item(),
                    )
            elif isinstance(last_layer, nn.Linear):
                for k, v in instance_output.items():
                    prob_v = nn.Softmax()(v)
                    max_prob_index = torch.argmax(prob_v).data.item()
                    instance_prob[k] = (
                        max_prob_index,
                        prob_v[max_prob_index].data.item(),
                    )
            else:
                raise RuntimeError("unsupported layer", type(last_layer))

        accuracy = sum(class_correct_count.values()) / \
            sum(class_count.values())
        return (
            loss,
            accuracy,
            {
                "per_class_accuracy": class_count,
                "per_sample_prob": instance_prob,
            },
        )
