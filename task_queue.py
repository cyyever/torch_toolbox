#!/usr/bin/env python
import queue
import threading
import traceback

import torch.multiprocessing

from .log import get_logger


class SentinelTask:
    pass


class TaskQueue(queue.Queue):
    def __init__(self, processor, worker_num=1):
        queue.Queue.__init__(self)
        self.worker_num = worker_num
        self.processor = processor
        self.threads = []
        self.stop_event = threading.Event()
        self.start()

    def start(self):
        self.stop()
        for _ in range(self.worker_num):
            t = threading.Thread(target=self.__worker, args=(self.stop_event,))
            self.threads.append(t)
            t.start()

    def stop(self):
        if not self.threads:
            return
        # stop workers
        for _ in range(self.worker_num):
            self.put(SentinelTask())
        # block until all tasks are done
        self.join()
        for thd in self.threads:
            thd.join()
        self.threads = None

    def force_stop(self):
        if not self.threads:
            return
        self.stop_event.set()
        # stop workers
        for _ in range(self.worker_num):
            self.put(SentinelTask())
        for thd in self.threads:
            thd.join()
        self.threads = None
        self.stop_event.clear()

    def add_task(self, task):
        self.put(task)

    def __worker(self, stop_event):
        while not stop_event.is_set():
            task = self.get()
            if isinstance(task, SentinelTask):
                self.task_done()
                break
            try:
                self.processor(task)
            except Exception as e:
                get_logger().error("catch exception:%s", e)
                get_logger().error("traceback:%s", traceback.format_exc())
            self.task_done()


def worker(q, processor_fun, stop_event, task_extra_args):
    while not stop_event.is_set():
        task = q.get()
        if isinstance(task, SentinelTask):
            break
        try:
            processor_fun(task, task_extra_args)
        except Exception as e:
            get_logger().error("catch exception:%s", e)
            get_logger().error("traceback:%s", traceback.format_exc())


class ProcessTaskQueue:
    def __init__(self, processor_fun, worker_num=1, task_extra_args=None):
        self.ctx = torch.multiprocessing.get_context("spawn")
        self.queue = self.ctx.Queue()
        self.worker_num = worker_num
        self.processor_fun = processor_fun
        self.processors = []
        self.stop_event = self.ctx.Event()
        self.task_extra_args = task_extra_args
        self.start()

    def start(self):
        self.stop()
        for _ in range(self.worker_num):
            t = self.ctx.Process(
                target=worker,
                args=(
                    self.queue,
                    self.processor_fun,
                    self.stop_event,
                    self.task_extra_args,
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
        self.processors = None

    def force_stop(self):
        if not self.processors:
            return
        self.stop_event.set()
        # stop workers
        for _ in range(self.worker_num):
            self.queue.put(SentinelTask())
        for thd in self.processors:
            thd.join()
        self.processors = None
        self.stop_event.clear()

    def add_task(self, task):
        self.queue.put(task)
