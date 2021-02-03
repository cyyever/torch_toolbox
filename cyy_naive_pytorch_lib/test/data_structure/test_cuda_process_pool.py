import torch

from data_structure.cuda_process_pool import CUDAProcessPool


def test_process_task_queue():
    pool = CUDAProcessPool()
    pool.exec(lambda: print("process is", torch.multiprocessing.current_process()))
    pool.stop()
