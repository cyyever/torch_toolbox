from inspect import signature

from cyy_naive_lib.algorithm.mapping_op import change_mapping_keys
from torchvision.models import MobileNetV2, resnet50
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.quantization.mobilenet import QuantizableMobileNetV2

from dataset import DatasetUtil
from dataset_collection import DatasetCollection
from ml_type import MachineLearningPhase, ModelType
from model_with_loss import ModelWithLoss
from models.densenet import DenseNet40
from models.lenet import LeNet5


def get_model(name: str, dataset_collection: DatasetCollection) -> ModelWithLoss:
    name_to_model_mapping: dict = {
        "LeNet5": LeNet5,
        "MobileNetV2": MobileNetV2,
        "QuantizableMobileNetV2": QuantizableMobileNetV2,
        "DenseNet40": DenseNet40,
        "ResNet50": resnet50,
        "FasterRCNN": fasterrcnn_resnet50_fpn,
    }
    name_to_model_mapping = change_mapping_keys(
        name_to_model_mapping, lambda x: x.lower()
    )
    model_constructor = name_to_model_mapping.get(name.lower(), None)
    if model_constructor is None:
        raise RuntimeError("unknown model name:", name)

    dataset_util = DatasetUtil(
        dataset_collection.get_dataset(MachineLearningPhase.Training)
    )
    sig = signature(model_constructor)
    kwargs = dict()
    for param in sig.parameters:
        if param in ("input_channels", "channels"):
            kwargs[param] = dataset_util.channel
        if param == "num_classes":
            label_num = dataset_util.get_label_number()
            if model_constructor is fasterrcnn_resnet50_fpn:
                label_num += 1
            kwargs[param] = label_num
        if param == "pretrained":
            kwargs[param] = False

    model_type = ModelType.Classification
    if model_constructor is fasterrcnn_resnet50_fpn:
        model_type = ModelType.Detection

    return ModelWithLoss(model_constructor(**kwargs), model_type=model_type)
