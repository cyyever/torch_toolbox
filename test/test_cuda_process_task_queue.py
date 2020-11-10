from cuda_process_task_queue import CUDAProcessTaskQueue


def hello(task, args):
    assert args
    return args


def test_process_task_queue():
    queue = CUDAProcessTaskQueue(hello)
    queue.start()
    queue.add_task(())
    devices = queue.get_result()
    assert devices
    queue.stop()
