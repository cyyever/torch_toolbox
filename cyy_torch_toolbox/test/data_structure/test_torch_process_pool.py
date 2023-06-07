from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.data_structure.torch_process_pool import \
    TorchProcessPool
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.dependency import has_torchvision
from cyy_torch_toolbox.ml_type import ExecutorHookPoint, StopExecutingException


def stop_training(*args, **kwargs):
    raise StopExecutingException()


def train(worker_id) -> None:
    if has_torchvision:
        get_logger().info("worker_id is %s", worker_id)
        trainer = DefaultConfig("MNIST", "LeNet5").create_trainer()
        trainer.hyper_parameter.epoch = 1
        trainer.hyper_parameter.learning_rate = 0.01
        trainer.append_named_hook(
            ExecutorHookPoint.AFTER_BATCH, "stop_training", stop_training
        )
        trainer.train()


def test_process_pool() -> None:
    pool = TorchProcessPool()
    for worker_id in range(2):
        pool.submit(train, worker_id)
    pool.shutdown()
