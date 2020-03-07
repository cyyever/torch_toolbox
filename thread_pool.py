import concurrent.futures
import threading
import traceback
from .log import get_logger


class ThreadPool:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.stopEvent = threading.Event()
        self.futures = []

    def stop(self):
        self.stopEvent.set()
        concurrent.futures.wait(self.futures)
        self.stopEvent.clear()

    def submit(self, loop_interval, fn, *args, **kwargs):
        def process():
            while not self.stopEvent.wait(loop_interval):
                try:
                    fn(*args, **kwargs)
                except Exception as e:
                    get_logger().error("catch exception:%s", e)
                    get_logger().error("traceback:%s", traceback.format_exc())

        self.futures.append(self.executor.submit(process))
