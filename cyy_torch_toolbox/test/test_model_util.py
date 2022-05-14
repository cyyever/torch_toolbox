from dataset_collection import (ClassificationDatasetCollection,
                                create_dataset_collection)
from model_factory import get_model, get_model_info
from model_util import ModelUtil


def test_model_info():
    models = get_model_info()
    assert models
    print(models.keys())


def test_get_submodules():
    mnist = create_dataset_collection(ClassificationDatasetCollection, "MNIST")
    model_util = ModelUtil(get_model("LeNet5", mnist).model)
    result = model_util.get_sub_modules()
    assert result
    print(result)


def test_get_blocks():
    mnist = create_dataset_collection(ClassificationDatasetCollection, "MNIST")
    model_util = ModelUtil(get_model("LeNet5", mnist).model)
    result = model_util.get_sub_module_blocks()
    assert result
