import copy
import torch

from .device import get_device, get_cpu_device
from .util import model_gradients_to_vector
from .hessian_vector_product import hessian_vector_product as _hessian_vector_product
from .dataset import get_class_count, dataset_with_indices


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

    def validate(self, batch_size, **kwargs):

        class_count = dict()
        class_correct_count = dict()

        per_class_accuracy = kwargs.get("per_class_accuracy", False)
        if per_class_accuracy:
            class_count = get_class_count(self.dataset)
            for k in class_count.keys():
                class_correct_count[k] = 0

        per_instance_loss = kwargs.get("per_instance_loss", False)
        per_instance_output = kwargs.get("per_instance_output", False)
        dataset = self.dataset
        if per_instance_loss or per_instance_output:
            dataset = dataset_with_indices(dataset)

        validation_data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size
        )

        use_grad = kwargs.get("use_grad", False)
        instance_validation_loss = dict()
        instance_output = dict()
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

                if per_instance_loss:
                    for i, instance_index in enumerate(batch[2]):
                        instance_index = instance_index.data.item()
                        instance_validation_loss[instance_index] = self.loss_fun(
                            outputs[i].unsqueeze(0), targets[i].unsqueeze(0))
                if per_instance_output:
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
                if use_grad:
                    loss2 = None
                    if loss_is_mean:
                        loss2 = loss * real_batch_size / len(dataset)
                    else:
                        loss2 = loss
                    loss2.backward()
                batch_loss = loss.data.item()
                if hasattr(self.loss_fun, "reduction") and (
                    self.loss_fun.reduction == "mean"
                    or self.loss_fun.reduction == "elementwise_mean"
                ):
                    batch_loss *= real_batch_size
                    batch_loss /= len(dataset)

                validation_loss += batch_loss
                correct = torch.eq(torch.max(outputs, dim=1)
                                   [1], targets).view(-1)

                if per_class_accuracy:
                    for k in class_count.keys():
                        class_correct_count[k] += torch.sum(
                            correct[targets == k]
                        ).item()

                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
                after_batch_callback = kwargs.get("after_batch_callback", None)
                if after_batch_callback is not None:
                    after_batch_callback(self.model, batch_loss)

            if per_class_accuracy:
                for k in class_count.keys():
                    class_count[k] = class_correct_count[k] / class_count[k]
            return (
                validation_loss,
                num_correct / num_examples,
                {
                    "per_class_accuracy": class_count,
                    "per_instance_loss": instance_validation_loss,
                    "per_instance_output": instance_output,
                },
            )

    def get_gradient(self):
        self.validate(64, use_grad=True)
        return model_gradients_to_vector(self.model)

    def hessian_vector_product(self, v, damping=0):
        res = None

        def after_batch_callback(model, batch_loss):
            nonlocal res
            if res is None:
                res = _hessian_vector_product(model, batch_loss, v)
            else:
                res += _hessian_vector_product(model, batch_loss, v)

        self.validate(
            64,
            use_grad=True,
            after_batch_callback=after_batch_callback)
        if damping != 0:
            res += damping * v
        return res
