from cyy_naive_lib.log import get_logger

from default_config import DefaultConfig
from data_structure.cuda_process_pool import CUDAProcessPool
from ml_type import StopExecutingException
from model_executor import ModelExecutorCallbackPoint


def stop_training(*args, **kwargs):
    raise StopExecutingException()


def train(worker_id):
    get_logger().info("worker_id is %s", worker_id)
    trainer = DefaultConfig("MNIST", "LeNet5").create_trainer()
    trainer.hyper_parameter.set_epoch(1)
    trainer.hyper_parameter.set_learning_rate(0.01)
    trainer.add_callback(ModelExecutorCallbackPoint.AFTER_BATCH, stop_training)
    trainer.train()


def test_process_task_queue():
    pool = CUDAProcessPool()
    for worker_id in range(2):
        pool.exec(train, worker_id)
    pool.stop()
