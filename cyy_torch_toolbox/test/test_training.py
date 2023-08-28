from cyy_torch_toolbox.default_config import Config
from cyy_torch_toolbox.dependency import (has_torch_geometric, has_torchtext,
                                          has_torchvision)
from cyy_torch_toolbox.device import DeviceGreedyAllocator
from cyy_torch_toolbox.ml_type import ExecutorHookPoint, StopExecutingException


def stop_training(*args, **kwargs):
    raise StopExecutingException()


def test_vision_training() -> None:
    if not has_torchvision:
        return
    config = Config(dataset_name="MNIST", model_name="LeNet5")
    config.trainer_config.hook_config.debug = True
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.01
    trainer = config.create_trainer()
    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "stop_training", stop_training
    )
    # trainer.model_with_loss.compile_model()
    trainer.train()


def test_text_training() -> None:
    if not has_torchtext:
        return
    device = DeviceGreedyAllocator().get_device(max_needed_bytes=9 * 1024 * 1024 * 1024)
    if device is None:
        return
    config = Config(dataset_name="IMDB", model_name="simplelstm")
    config.trainer_config.hook_config.debug = True
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.01
    config.dc_config.dataset_kwargs["tokenizer"] = {"type": "spacy"}
    trainer = config.create_trainer()
    # trainer.model_with_loss.compile_model()
    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "stop_training", stop_training
    )
    trainer.train()


def test_graph_training() -> None:
    if not has_torch_geometric:
        return
    config = Config(dataset_name="Yelp", model_name="OneGCN")
    config.trainer_config.hook_config.debug = True
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.01
    trainer = config.create_trainer()
    # trainer.model_with_loss.compile_model()
    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "stop_training", stop_training
    )
    trainer.train()
