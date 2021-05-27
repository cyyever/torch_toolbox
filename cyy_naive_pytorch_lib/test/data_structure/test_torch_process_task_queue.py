from shutil import which

from data_structure.torch_process_task_queue import TorchProcessTaskQueue


def hello(task, args):
    assert args
    return args


def test_process_task_queue():
    if which("nvcc"):
        queue = TorchProcessTaskQueue(hello)
        queue.start()
        queue.add_task(())
        devices = queue.get_result()
        assert devices
        queue.stop()
        queue = TorchProcessTaskQueue(hello, use_manager=True)
        queue.start()
        queue.add_task(())
        devices = queue.get_result()
        assert devices
        queue.stop()
