import torch

from cyy_naive_lib.mapping_op import change_mapping_keys
from torchvision.models import MobileNetV2

from dataset import DatasetUtil
from model_loss import ModelWithLoss
from models.lenet import LeNet5
from models.densenet import DenseNet40


def get_model(
        name: str,
        dataset: torch.utils.data.Dataset = None) -> ModelWithLoss:
    name_to_model_mapping: dict = {
        "LeNet5": LeNet5,
        "MobileNet": MobileNetV2,
        "DenseNet40": DenseNet40,
    }
    change_mapping_keys(name_to_model_mapping, lambda x: x.lower())
    model_constructor = name_to_model_mapping.get(name.lower(), None)
    if model_constructor is None:
        raise RuntimeError("unknown model name:", name)

    dataset_util = DatasetUtil(dataset)
    kwargs = dict()
    kwargs["input_channels"] = dataset_util.channel
    kwargs["num_classes"] = dataset_util.get_label_number()
    kwargs["channels"] = dataset_util.channel
    return ModelWithLoss(model_constructor(**kwargs))
