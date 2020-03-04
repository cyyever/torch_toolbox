#!/usr/bin/env python
import queue
import threading
import traceback
from .log import get_logger


class TaskQueue(queue.Queue):
    def __init__(self, processor, worker_num=10):
        queue.Queue.__init__(self)
        self.worker_num = worker_num
        self.processor = processor
        self.threads = []
        self.stop_event = threading.Event()
        for _ in range(self.worker_num):
            t = threading.Thread(target=self.__worker, args=(self.stop_event,))
            self.threads.append(t)
            t.start()

    def stop(self):
        # stop workers
        for _ in range(self.worker_num):
            self.put(None)
        # block until all tasks are done
        self.join()
        for thd in self.threads:
            thd.join()

    def force_stop(self):
        self.stop_event.set()
        # stop workers
        for _ in range(self.worker_num):
            self.put(None)
        for thd in self.threads:
            thd.join()

    def add_task(self, task):
        self.put(task)

    def __worker(self, stop_event):
        while True:
            if stop_event.wait(0.00001):
                break
            task = None
            try:
                task = self.get(block=False, timeout=1)
            except queue.Empty:
                continue
            if task is None:
                self.task_done()
                break
            try:
                self.processor(task)
            except Exception as e:
                get_logger().error("catch exception:%s", e)
                get_logger().error("traceback:%s", traceback.format_exc())
            self.task_done()
