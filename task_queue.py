#!/usr/bin/env python
import queue
import threading
import traceback
from .log import get_logger


class TaskQueue(queue.Queue):
    class SentinelTask:
        pass

    def __init__(self, processor, worker_num=10):
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
            self.put(TaskQueue.SentinelTask())
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
            self.put(TaskQueue.SentinelTask())
        for thd in self.threads:
            thd.join()
        self.threads = None
        self.stop_event.clear()

    def add_task(self, task):
        self.put(task)

    def __worker(self, stop_event):
        while not stop_event.is_set():
            task = self.get()
            if isinstance(task, TaskQueue.SentinelTask):
                self.task_done()
                break
            try:
                self.processor(task)
            except Exception as e:
                get_logger().error("catch exception:%s", e)
                get_logger().error("traceback:%s", traceback.format_exc())
            self.task_done()
