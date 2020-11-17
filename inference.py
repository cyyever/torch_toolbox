import copy
from typing import Optional

import torch
import torch.nn as nn

from hyper_parameter import HyperParameter
from device import get_device, put_data_to_device
from model_loss import ModelWithLoss
from model_util import ModelUtil
from util import get_batch_size
from dataset import DatasetUtil


class Inferencer:
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset,
        hyper_parameter: Optional[HyperParameter] = None,
    ):
        self.__model_with_loss = copy.deepcopy(model_with_loss)
        self.__dataset = dataset
        self.__hyper_parameter = hyper_parameter

    @property
    def model_with_loss(self):
        return self.__model_with_loss

    @property
    def model(self):
        return self.model_with_loss.model

    @property
    def loss_fun(self):
        return self.model_with_loss.loss_fun

    def set_dataset(self, dataset):
        self.__dataset = dataset

    def load_model(self, model_path):
        self.model_with_loss.set_model(
            torch.load(model_path, map_location=get_device())
        )

    def inference(self, **kwargs):
        class_count = dict()
        class_correct_count = dict()

        per_class_accuracy = kwargs.get("per_class_accuracy", False)
        if per_class_accuracy:
            for k in range(DatasetUtil(self.__dataset).get_label_number()):
                class_correct_count[k] = 0

        per_sample_output = kwargs.get("per_sample_output", False)
        per_sample_prob = kwargs.get("per_sample_prob", False)
        if per_sample_prob:
            per_sample_output = True
        validation_data_loader = self.__hyper_parameter.get_dataloader(
            self.__dataset, False
        )

        use_grad = kwargs.get("use_grad", False)
        instance_output = dict()
        instance_prob = dict()
        with torch.set_grad_enabled(use_grad):
            num_correct = 0
            num_examples = 0
            device = get_device()
            self.model.eval()
            self.model.zero_grad()
            self.model.to(device)
            total_loss = torch.zeros(1)
            total_loss = total_loss.to(device)
            for batch in validation_data_loader:
                inputs = put_data_to_device(batch[0], device)
                targets = put_data_to_device(batch[1], device)
                real_batch_size = get_batch_size(inputs)

                result = self.model_with_loss(
                    inputs, targets, for_training=False)
                print("result is ", result)
                batch_loss = result["loss"]
                outputs = result.get("output", None)
                # outputs = self.model(inputs)

                if per_sample_output:
                    # outputs = result["output"]
                    for i, instance_index in enumerate(batch[2]):
                        instance_index = instance_index.data.item()
                        instance_output[instance_index] = outputs[i]

                # batch_loss = self.loss_fun(outputs, targets)
                if self.model_with_loss.is_averaged_loss():
                    normalized_batch_loss = (
                        batch_loss * real_batch_size / len(self.__dataset)
                    )
                else:
                    normalized_batch_loss = batch_loss
                if use_grad:
                    normalized_batch_loss.backward()

                total_loss += normalized_batch_loss
                correct = torch.eq(torch.max(outputs, dim=1)
                                   [1], targets).view(-1)

                if per_class_accuracy:
                    for k in class_count:
                        class_correct_count[k] += torch.sum(
                            correct[targets == k]
                        ).item()

                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]

            if per_class_accuracy:
                for k in class_count:
                    class_count[k] = class_correct_count[k] / class_count[k]
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

            return (
                total_loss,
                num_correct / num_examples,
                {
                    "per_class_accuracy": class_count,
                    "per_sample_output": instance_output,
                    "per_sample_prob": instance_prob,
                },
            )

    def get_gradient(self):
        self.inference(use_grad=True)
        return ModelUtil(self.model).get_gradient_list()
