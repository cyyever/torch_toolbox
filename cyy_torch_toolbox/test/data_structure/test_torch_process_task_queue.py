from shutil import which

from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue


def hello(task, args):
    assert not task
    assert args
    return 1


def test_process_task_queue():
    if which("nvcc"):
        queue = TorchProcessTaskQueue(hello)
        queue.start()
        queue.add_task(())
        res = queue.get_result()
        assert res == 1
        queue.stop()
