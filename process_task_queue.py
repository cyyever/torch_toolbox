#!/usr/bin/env python
import traceback

import torch.multiprocessing

from log import get_logger
from device import get_cuda_devices


class SentinelTask:
    pass


def worker(
        task_queue,
        result_queue,
        processor_fun,
        stop_event,
        extra_arguments: list):
    while not stop_event.is_set():
        task = task_queue.get()
        if isinstance(task, SentinelTask):
            break
        try:
            res = processor_fun(task, extra_arguments)
            if res is not None:
                result_queue.put(res)
        except Exception as e:
            get_logger().error("catch exception:%s", e)
            get_logger().error("traceback:%s", traceback.format_exc())


class ProcessTaskQueue:
    def __init__(self, processor_fun, worker_num=1):
        self.ctx = torch.multiprocessing.get_context("spawn")
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.worker_num = worker_num
        self.processor_fun = processor_fun
        self.processors = dict()
        self.stop_event = self.ctx.Event()
        self.start()

    def start(self):
        for worker_id in range(len(self.processors), self.worker_num):
            t = self.ctx.Process(
                target=worker,
                args=(
                    self.task_queue,
                    self.result_queue,
                    self.processor_fun,
                    self.stop_event,
                    self._get_extra_task_arguments(worker_id),
                ),
            )
            self.processors[worker_id] = t
            t.start()

    def join(self):
        for processor in self.processors.values():
            processor.join()

    def stop(self):
        if not self.processors:
            return
        # stop workers
        for _ in range(self.worker_num):
            self.add_task(SentinelTask())
        # block until all tasks are done
        self.join()
        self.processors = dict()

    def force_stop(self):
        self.stop_event.set()
        self.stop()
        self.stop_event.clear()

    def add_task(self, task):
        self.task_queue.put(task)

    def get_result(self):
        return self.result_queue.get()

    def _get_extra_task_arguments(self, worker_id):
        return []


class CUDAProcessTaskQueue(ProcessTaskQueue):
    def __init__(self, processor_fun):
        self.cuda_devices = get_cuda_devices()
        super().__init__(processor_fun, len(self.cuda_devices))

    def _get_extra_task_arguments(self, worker_id):
        return [self.cuda_devices[worker_id]]
