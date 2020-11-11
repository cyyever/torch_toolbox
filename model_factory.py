import torch

from cyy_naive_lib.log import get_logger
from cyy_naive_lib.mapping_op import change_mapping_keys

from models.lenet import LeNet5
from torchvision.models import MobileNetV2
from model_loss import ModelWithLoss
from models.densenet2 import (
    densenet_CIFAR10,
    densenet_CIFAR10_group_norm,
)
from models.senet.se_resnet_group_norm import se_resnet20_group_norm


def get_model(name: str, dataset: torch.utils.data.Dataset = None):
    name_to_model_mapping: dict = {
        "LeNet5": LeNet5,
        "MobileNet": MobileNetV2,
    }
    change_mapping_keys(name_to_model_mapping, lambda x: x.lower())
    model_constructor = name_to_model_mapping.get(name.lower(), None)
    if model_constructor is None:
        get_logger().error("unknown model name %s", name)
        raise RuntimeError("unknown model name:", name)
    return model_constructor()
