from shutil import which

from data_structure.torch_process_task_queue import TorchProcessTaskQueue


def hello(task, args):
    assert not task
    assert args
    return 1


def test_process_task_queue():
    if which("nvcc"):
        queue = TorchProcessTaskQueue(hello, use_manager=True)
        queue.start()
        queue.add_task(())
        res = queue.get_result()
        assert res == 1
        queue.stop()
