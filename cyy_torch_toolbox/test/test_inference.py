from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.dependency import has_torchtext, has_torchvision
from cyy_torch_toolbox.ml_type import MachineLearningPhase


def test_inference() -> None:
    if not has_torchvision:
        return
    config = DefaultConfig(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 1
    trainer = config.create_trainer()
    inferencer = trainer.get_inferencer(MachineLearningPhase.Test)
    inferencer.inference()


def test_gradient() -> None:
    if not has_torchtext:
        return
    config = DefaultConfig(dataset_name="IMDB", model_name="simplelstm")
    config.hyper_parameter_config.epoch = 1
    config.dc_config.dataset_kwargs["tokenizer"] = {"type": "spacy"}
    trainer = config.create_trainer()
    inferencer = trainer.get_inferencer(MachineLearningPhase.Test)
    inferencer.get_gradient()
