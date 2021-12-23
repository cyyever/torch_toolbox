from dataset_collection import DatasetCollection
from model_factory import get_model
from model_util import ModelUtil


def test_get_submodules():
    mnist = DatasetCollection.get_by_name("MNIST")
    model_util = ModelUtil(get_model("LeNet5", mnist).model)
    model_util.get_sub_modules()
