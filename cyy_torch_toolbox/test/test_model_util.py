from cyy_torch_toolbox.dataset_collection import (
    ClassificationDatasetCollection, create_dataset_collection)
from cyy_torch_toolbox.model_factory import get_model, get_model_info


def test_model_info():
    models = get_model_info()
    assert models


def test_get_module():
    mnist = create_dataset_collection(ClassificationDatasetCollection, "MNIST")
    get_model("LeNet5", mnist)
