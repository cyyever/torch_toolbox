import torch.nn as nn
from dataset_collection import DatasetCollection
from model_factory import get_model
from model_util import ModelUtil


def test_get_submodules():
    mnist = DatasetCollection.get_by_name("MNIST")
    model_util = ModelUtil(get_model("LeNet5", mnist).model)
    model_util.get_sub_module_blocks()
    model_util.remove_statistical_variables()
    # num1 = len(model_util.get_sub_modules())
    # assert model_util.remove_sub_modules(module_classes={nn.Linear})
    # num2 = len(model_util.get_sub_modules())
    # assert num1 != num2
