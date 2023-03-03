from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.data_structure.torch_process_pool import \
    TorchProcessPool
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)


def stop_training(*args, **kwargs):
    raise StopExecutingException()


def train(worker_id):
    get_logger().info("worker_id is %s", worker_id)
    trainer = DefaultConfig("MNIST", "LeNet5").create_trainer()
    trainer.hyper_parameter.set_epoch(1)
    trainer.hyper_parameter.set_learning_rate(0.01)
    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "stop_training", stop_training
    )
    trainer.train()


def test_process_pool():
    pool = TorchProcessPool()
    for worker_id in range(2):
        pool.exec(train, worker_id)
    pool.shutdown()
