import torch

from cyy_naive_lib.mapping_op import change_mapping_keys
from torchvision.models import MobileNetV2

from inspect import signature
from dataset import DatasetUtil
from model_loss import ModelWithLoss
from models.lenet import LeNet5
from models.densenet import DenseNet40


def get_model(name: str, dataset: torch.utils.data.Dataset) -> ModelWithLoss:
    name_to_model_mapping: dict = {
        "LeNet5": LeNet5,
        "MobileNet": MobileNetV2,
        "DenseNet40": DenseNet40,
    }
    name_to_model_mapping = change_mapping_keys(
        name_to_model_mapping, lambda x: x.lower()
    )
    model_constructor = name_to_model_mapping.get(name.lower(), None)
    if model_constructor is None:
        raise RuntimeError("unknown model name:", name)

    dataset_util = DatasetUtil(dataset)
    sig = signature(model_constructor)
    kwargs = dict()
    for param in sig.parameters:
        if param in ("input_channels", "channels"):
            kwargs[param] = dataset_util.channel
        if param in ("num_classes"):
            kwargs[param] = dataset_util.get_label_number()

    return ModelWithLoss(model_constructor(**kwargs))
