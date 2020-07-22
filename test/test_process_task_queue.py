from process_task_queue import CUDAProcessTaskQueue


def hello(task, args):
    print("hello world")
    assert task == ()
    print(args)


def test_process_task_queue():
    queue = CUDAProcessTaskQueue(hello)
    queue.start()
    queue.add_task(())
    queue.stop()
