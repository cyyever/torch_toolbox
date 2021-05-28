from model_util import ModelUtil
from models.lenet import LeNet5


def test_get_submodules():
    model_util = ModelUtil(LeNet5())
    model_util.get_sub_modules()
