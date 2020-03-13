import copy
import torch

from .device import get_device
from .log import get_logger
from .util import model_gradients_to_vector
from .hessian_vector_product import hessian_vector_product as _hessian_vector_product
from .dataset import get_class_count


class Validator:
    def __init__(
        self, model, loss_fun, validation_dataset,
    ):
        self.model = copy.deepcopy(model)
        self.loss_fun = loss_fun
        self.validation_dataset = validation_dataset

    def validate(
        self,
        batch_size,
        use_grad=False,
        per_class_accuracy=False,
        after_batch_callback=None,
    ):

        class_count = dict()
        class_correct_count = dict()
        if per_class_accuracy:
            class_count = get_class_count(self.validation_dataset)
            for k in class_count.keys():
                class_correct_count[k] = 0

        validation_data_loader = torch.utils.data.DataLoader(
            self.validation_dataset, batch_size=batch_size
        )
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
                loss = self.loss_fun(outputs, targets)
                if use_grad:
                    loss.backward()
                batch_loss = loss.data.item()
                if hasattr(self.loss_fun, "reduction") and (
                    self.loss_fun.reduction == "mean"
                    or self.loss_fun.reduction == "elementwise_mean"
                ):
                    batch_loss *= real_batch_size
                    batch_loss /= len(self.validation_dataset)
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
                if after_batch_callback:
                    after_batch_callback(self.model, batch_loss)

            if per_class_accuracy:
                for k in class_count.keys():
                    class_count[k] = class_correct_count[k] / class_count[k]
            return (validation_loss, num_correct / num_examples, class_count)

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
