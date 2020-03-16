from enum import Enum, auto
import shutil
import os
import time
import queue
import collections

import torch

from .task_queue import TaskQueue
from .log import get_logger


class Action(Enum):
    WRITE = auto()
    READ = auto()


class ActionResult(Enum):
    WRITE_SUCC = auto()
    WRITE_FAILED = auto()
    READ_SUCC = auto()
    READ_FAILED = auto()


class DataInfo(Enum):
    IN_MEMORY = auto()
    IN_MEMORY_NEW_DATA = auto()
    IN_DISK = auto()
    SAVING = auto()
    LOADING = auto()


class LargeDict:
    def __init__(self, storage_dir=None):
        self.__in_memory_key_number = 128
        self.data = dict()
        self.data_info = dict()
        self.__LRU_keys = collections.OrderedDict()
        self.__LRU_keys_cnt = 0
        self.storage_dir = None
        self.permanent = False

        if storage_dir is not None:
            self.set_storage_dir(storage_dir)
            self.data_info = {
                int(f): DataInfo.IN_DISK
                for f in os.listdir(storage_dir)
                if os.path.isfile(os.path.join(storage_dir, f))
            }
            self.permanent = True

        self.io_respose_queue = queue.Queue()
        self.io_request_queue = TaskQueue(LargeDict.do_io, 1)
        self.flush_all_once = False

    @staticmethod
    def do_io(task):
        action, response_queue, data = task
        if action == Action.WRITE:
            results = dict()
            for k, v in data.items():
                item_path, data = v
                result = ActionResult.WRITE_SUCC
                try:
                    torch.save(data, item_path)
                except Exception:
                    get_logger().error("save key %s failed", k)
                    result = ActionResult.WRITE_FAILED
                results[k] = (result, None)
            response_queue.put(results)
            return
        if action == Action.READ:
            results = dict()
            for k, path in data.items():
                value = None
                result = ActionResult.READ_SUCC
                try:
                    value = torch.load(path)
                except Exception:
                    get_logger().error("load key %s failed", k)
                    result = ActionResult.READ_FAILED
                results[k] = (result, value)
            response_queue.put(results)
            return

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def set_in_memory_key_number(self, num):
        self.__in_memory_key_number = num

    def set_permanent(self):
        self.permanent = True

    def flush_all(self):
        self.flush_all_once = True
        self.__flush()

    def flush(self):
        self.__flush()

    def print_data_info(self):
        cnt = {}
        for k, v in self.data_info.items():
            if v not in cnt:
                cnt[v] = 1
            else:
                cnt[v] += 1
        for k, v in cnt.items():
            get_logger().debug("%s => %s", k, v)

    def get_in_memeory_cnt(self):
        cnt = 0
        for v in self.data_info.values():
            if v in (DataInfo.IN_MEMORY, DataInfo.IN_MEMORY_NEW_DATA,):
                cnt += 1
        return cnt

    def keys(self):
        return self.data_info.keys()

    def prefetch(self, keys, ignore_unknown_keys=True):
        self.__process_io_response(block=False)
        result = []
        io_keys = dict()
        for key in keys:
            data_info = self.data_info.get(key, None)
            if data_info is None:
                if ignore_unknown_keys:
                    continue
                raise KeyError(key)
            if data_info == DataInfo.LOADING:
                continue
            if key in self.data:
                if data_info != DataInfo.IN_MEMORY_NEW_DATA:
                    self.data_info[key] = DataInfo.IN_MEMORY

                self.__update_item_access_time(key)
                result = [self.data[key]]
                continue
            self.data_info[key] = DataInfo.LOADING
            io_keys[key] = self.get_key_storage_path(key)

        if io_keys:
            self.io_request_queue.add_task(
                (Action.READ, self.io_respose_queue, io_keys)
            )
        return result

    def release(self):
        get_logger().info("release data structure")
        if self.permanent:
            self.flush_all()
        self.io_request_queue.stop()
        self.data = None

        if (
            not self.permanent
            and self.storage_dir is not None
            and os.path.exists(self.storage_dir)
        ):
            get_logger().debug("rmtree %s", self.storage_dir)
            shutil.rmtree(self.storage_dir)

    def get_key_storage_path(self, key):
        if self.storage_dir is None:
            raise RuntimeError("no storage_dir")
        return os.path.join(self.storage_dir, str(key))

    def __len__(self):
        return len(self.keys())

    def __contains__(self, key):
        return key in self.data_info

    def __getitem__(self, key):
        # cur_time = time.time()
        result = self.prefetch([key])
        if result:
            # get_logger().debug("get use time %s", (time.time() - cur_time) * 1000)
            return result[0]

        while True:
            self.__process_io_response(block=True)
            if self.data_info[key] in (
                DataInfo.IN_MEMORY,
                DataInfo.IN_MEMORY_NEW_DATA,
            ):
                get_logger().warning("read key %s from disk", key)
                # get_logger().error("get use time %s", (time.time() - cur_time) * 1000)
                return self.data[key]
        raise KeyError(key)

    def __setitem__(self, key, val):
        self.data[key] = val
        self.__add_item_access_time(key)
        self.data_info[key] = DataInfo.IN_MEMORY_NEW_DATA
        # cur_time = time.time()
        self.__flush()
        # get_logger().error("set use time %s", (time.time() - cur_time) * 1000)

    def __delitem__(self, key):
        data_info = self.data_info.pop(key, None)
        if data_info is None:
            raise KeyError(key)
        self.data.pop(key, None)
        self.__LRU_keys.pop(key, None)
        self.__LRU_keys_cnt -= 1
        shutil.rmtree(self.get_key_storage_path(key))

    def __del__(self):
        self.release()

    def __get_in_memory_key_number(self):
        if self.flush_all_once:
            return 0
        return self.__in_memory_key_number

    def __change_data_info(self, key, from_info, to_info):
        data_info = self.data_info.pop(key, None)
        if data_info is None:
            get_logger().warning("no key %s,ignore", key)
            return False
        if data_info != from_info:
            get_logger().warning("canceled key %s, info is %s", key, data_info)
            return False
        self.data_info[key] = to_info
        return True

    def __could_flush(self):
        return self.__LRU_keys_cnt > self.__get_in_memory_key_number() * 2

    def __has_expired_items(self):
        return self.__LRU_keys_cnt > self.__get_in_memory_key_number()

    def __get_expired_items(self):
        io_items = dict()
        while self.__has_expired_items():
            key = self.__LRU_keys.popitem(last=False)[0]
            self.__LRU_keys_cnt -= 1
            data_info = self.data_info.get(key, None)
            assert data_info in (
                DataInfo.IN_MEMORY, DataInfo.IN_MEMORY_NEW_DATA,)
            if data_info is None:
                continue
            if data_info == DataInfo.IN_MEMORY_NEW_DATA:
                self.data_info[key] = DataInfo.SAVING
                io_items[key] = (
                    self.get_key_storage_path(key),
                    self.data[key])
            else:
                self.data_info[key] = DataInfo.IN_DISK
                self.data.pop(key)
        self.flush_all_once = False
        return io_items

    def __update_item_access_time(self, key):
        self.__LRU_keys.move_to_end(key)

    def __add_item_access_time(self, key):
        self.__LRU_keys[key] = None
        self.__LRU_keys_cnt += 1

    def __flush(self):
        if self.__could_flush():
            items = self.__get_expired_items()
            self.io_request_queue.add_task(
                (Action.WRITE, self.io_respose_queue, items))
            get_logger().error("do_flush")

    def __process_io_response(self, block=False):
        results = None
        try:
            results = self.io_respose_queue.get(block=block)
        except queue.Empty:
            return
        for key, v in results.items():
            result, data = v
            data_info = self.data_info.get(key, None)
            if result in (ActionResult.WRITE_SUCC, ActionResult.WRITE_FAILED):
                if data_info != DataInfo.SAVING:
                    continue
            if result == ActionResult.WRITE_SUCC:
                self.data_info[key] = DataInfo.IN_DISK
                self.data.pop(key)
                continue
            if result == ActionResult.WRITE_FAILED:
                self.__add_item_access_time(key)
                self.data_info[key] = DataInfo.IN_MEMORY_NEW_DATA
                continue

            if result in (ActionResult.READ_SUCC, ActionResult.READ_FAILED):
                if data_info != DataInfo.LOADING:
                    continue
            if result == ActionResult.READ_SUCC:
                self.data[key] = data
                self.__add_item_access_time(key)
                self.data_info[key] = DataInfo.IN_MEMORY
                continue
            if result == ActionResult.READ_FAILED:
                raise KeyError(key)
