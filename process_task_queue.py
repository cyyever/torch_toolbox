#!/usr/bin/env python
import traceback

import torch.multiprocessing

from log import get_logger
from device import get_cuda_devices


class SentinelTask:
    pass


def worker(q, processor_fun, stop_event, extra_arguments: list):
    while not stop_event.is_set():
        task = q.get()
        if isinstance(task, SentinelTask):
            break
        try:
            print(extra_arguments)
            processor_fun(task, extra_arguments)
        except Exception as e:
            get_logger().error("catch exception:%s", e)
            get_logger().error("traceback:%s", traceback.format_exc())


class ProcessTaskQueue:
    def __init__(self, processor_fun, worker_num=1):
        self.ctx = torch.multiprocessing.get_context("spawn")
        self.queue = self.ctx.Queue()
        self.worker_num = worker_num
        self.processor_fun = processor_fun
        self.processors = []
        self.stop_event = self.ctx.Event()
        self.start()

    def start(self):
        self.stop()
        for worker_id in range(self.worker_num):
            t = self.ctx.Process(
                target=worker,
                args=(
                    self.queue,
                    self.processor_fun,
                    self.stop_event,
                    self._get_extra_task_arguments(worker_id),
                ),
            )
            self.processors.append(t)
            t.start()

    def stop(self):
        if not self.processors:
            return
        # stop workers
        for _ in range(self.worker_num):
            self.queue.put(SentinelTask())
        # block until all tasks are done
        for processor in self.processors:
            processor.join()
        self.processors = []

    def force_stop(self):
        self.stop_event.set()
        self.stop()
        self.stop_event.clear()

    def add_task(self, task):
        self.queue.put(task)

    def _get_extra_task_arguments(self, worker_id):
        return []


class CUDAProcessTaskQueue(ProcessTaskQueue):
    def __init__(self, processor_fun):
        self.cuda_devices = get_cuda_devices()
        super().__init__(processor_fun, len(self.cuda_devices))

    def _get_extra_task_arguments(self, worker_id):
        return [self.cuda_devices[worker_id]]
