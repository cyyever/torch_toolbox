from cyy_torch_toolbox.dataset_collection import (
    ClassificationDatasetCollection, create_dataset_collection)
from cyy_torch_toolbox.model_factory import get_model, get_model_info
from cyy_torch_toolbox.model_util import ModelUtil


def test_model_info():
    models = get_model_info()
    assert models


def test_get_modules():
    mnist = create_dataset_collection(ClassificationDatasetCollection, "MNIST")
    model_util = ModelUtil(get_model("LeNet5", mnist).model)
    result = model_util.get_modules()
    assert result
