from cyy_torch_toolbox.dataset_collection import (
    ClassificationDatasetCollection, create_dataset_collection)
from cyy_torch_toolbox.model_factory import get_model, get_model_info
from cyy_torch_toolbox.model_util import ModelUtil


def test_model_info():
    models = get_model_info()
    assert models


def test_get_submodules():
    mnist = create_dataset_collection(ClassificationDatasetCollection, "MNIST")
    model_util = ModelUtil(get_model("LeNet5", mnist).model)
    result = model_util.get_sub_modules()
    assert result


def test_get_blocks():
    # mnist = create_dataset_collection(ClassificationDatasetCollection, "MNIST")
    # model_util = ModelUtil(get_model("LeNet5", mnist).model)
    # result = model_util.get_sub_module_blocks()
    # print(result)
    # assert result
    cifar10 = create_dataset_collection(ClassificationDatasetCollection, "CIFAR10")
    model_util = ModelUtil(get_model("Densenet40", cifar10).model)
    result = model_util.get_sub_module_blocks()
    print(result)
    assert result
