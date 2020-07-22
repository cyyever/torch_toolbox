import copy
import torch

from device import get_device, get_cpu_device
from model_util import ModelUtil
from dataset import get_class_count, dataset_with_indices


class Validator:
    def __init__(
        self, model, loss_fun, dataset,
    ):
        try:
            self.model = copy.deepcopy(model)
        except RuntimeError:
            self.model = model
        self.loss_fun = loss_fun
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def validate(self, batch_size, **kwargs):

        class_count = dict()
        class_correct_count = dict()

        per_class_accuracy = kwargs.get("per_class_accuracy", False)
        if per_class_accuracy:
            class_count = get_class_count(self.dataset)
            for k in class_count:
                class_correct_count[k] = 0

        per_sample_loss = kwargs.get("per_sample_loss", False)
        per_sample_output = kwargs.get("per_sample_output", False)
        dataset = dataset_with_indices(self.dataset)

        validation_data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size
        )

        use_grad = kwargs.get("use_grad", False)
        instance_validation_loss = dict()
        instance_output = dict()
        instance_prob = dict()
        with torch.set_grad_enabled(use_grad):
            num_correct = 0
            num_examples = 0
            device = get_device()
            self.model.eval()
            self.model.zero_grad()
            self.model.to(device)
            validation_loss = torch.zeros(1)
            validation_loss = validation_loss.to(device)
            for batch in validation_data_loader:
                real_batch_size = batch[0].shape[0]
                inputs = batch[0]
                targets = batch[1]
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = self.model(inputs)

                if per_sample_loss:
                    for i, instance_index in enumerate(batch[2]):
                        instance_index = instance_index.data.item()
                        instance_validation_loss[instance_index] = self.loss_fun(
                            outputs[i].unsqueeze(0), targets[i].unsqueeze(0))
                if per_sample_output:
                    for i, instance_index in enumerate(batch[2]):
                        instance_index = instance_index.data.item()
                        instance_output[instance_index] = outputs[i].to(
                            get_cpu_device()
                        )

                loss_is_mean = False
                if hasattr(self.loss_fun, "reduction") and (
                    self.loss_fun.reduction == "mean"
                    or self.loss_fun.reduction == "elementwise_mean"
                ):
                    loss_is_mean = True

                loss = self.loss_fun(outputs, targets)
                if loss_is_mean:
                    batch_loss = loss * real_batch_size / len(dataset)
                else:
                    batch_loss = loss
                if use_grad:
                    batch_loss.backward()
                batch_loss = loss.data.item()
                if loss_is_mean:
                    batch_loss *= real_batch_size
                    batch_loss /= len(dataset)

                validation_loss += batch_loss
                correct = torch.eq(torch.max(outputs, dim=1)
                                   [1], targets).view(-1)

                if per_class_accuracy:
                    for k in class_count:
                        class_correct_count[k] += torch.sum(
                            correct[targets == k]
                        ).item()

                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
                after_batch_callback = kwargs.get("after_batch_callback", None)
                if after_batch_callback is not None:
                    after_batch_callback(self.model, batch_loss)

            if per_class_accuracy:
                for k in class_count:
                    class_count[k] = class_correct_count[k] / class_count[k]
            if instance_output:
                for k, v in instance_output.items():
                    max_prob_index = torch.argmax(v).data.item()
                    instance_prob[k] = (
                        max_prob_index,
                        v[max_prob_index].exp().data.item(),
                    )

            return (
                validation_loss,
                num_correct / num_examples,
                {
                    "per_class_accuracy": class_count,
                    "per_sample_loss": instance_validation_loss,
                    "per_sample_output": instance_output,
                    "per_sample_prob": instance_prob,
                },
            )

    def get_gradient(self):
        self.validate(64, use_grad=True)
        return ModelUtil(self.model).get_gradient_list()
