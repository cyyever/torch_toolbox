from enum import Enum, auto
import shutil
import os
import time
import threading
import collections

import torch

from .task_queue import TaskQueue
from .log import get_logger


class DataInfo(Enum):
    IN_MEMORY = auto()
    IN_MEMORY_NEW_DATA = auto()
    IN_DISK = auto()
    PRE_SAVING = auto()
    SAVING = auto()
    PRE_LOAD = auto()
    LOADING = auto()
    PRE_DELETE = auto()


class LargeDict:
    def __init__(self, storage_dir=None):
        self.in_memory_key_number = 128
        self.lock = threading.RLock()
        self.data = dict()
        self.data_info = dict()
        self.LRU_keys = collections.OrderedDict()
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

        self.write_queue = TaskQueue(LargeDict.write_item, 1)
        self.delete_queue = TaskQueue(LargeDict.delete_item, 1)
        self.fetch_queue = TaskQueue(LargeDict.read_item, 1)
        self.flush_thread = TaskQueue(LargeDict.flush_item, 1)
        self.flush_all_once = False
        self.fetch_event = threading.Event()
        self.wait_fetch_event = False

    @staticmethod
    def flush_item(task):
        large_dict = task
        while True:
            with large_dict.lock:
                succ_flag, key = large_dict.__pop_expired_access_time()
                if not succ_flag:
                    return
                assert large_dict.data_info[key] in (
                    DataInfo.IN_MEMORY,
                    DataInfo.IN_MEMORY_NEW_DATA,
                )
                if large_dict.data_info[key] == DataInfo.IN_MEMORY_NEW_DATA:
                    large_dict.data_info[key] = DataInfo.PRE_SAVING
                    large_dict.write_queue.add_task((large_dict, key))
                else:
                    large_dict.data_info[key] = DataInfo.IN_DISK
                    large_dict.data.pop(key)

    @staticmethod
    def write_item(task):
        large_dict, key = task
        value = None
        with large_dict.lock:
            if not large_dict.__change_data_info(
                key, DataInfo.PRE_SAVING, DataInfo.SAVING
            ):
                return
            value = large_dict.data[key]

        item_path = large_dict.get_key_storage_path(key)
        succ_flag = True
        try:
            torch.save(value, item_path)
        except Exception:
            get_logger().error("save key %s failed,keep it in memory", key)
            succ_flag = False

        with large_dict.lock:
            if key not in large_dict.data_info:
                if os.path.exists(item_path):
                    shutil.rmtree(item_path)
                return
            if large_dict.data_info[key] != DataInfo.SAVING:
                get_logger().warning(
                    "canceled key %s, info is %s",
                    key,
                    large_dict.data_info[key])
                return
            if not succ_flag:
                large_dict.data_info[key] = DataInfo.IN_MEMORY_NEW_DATA
                large_dict.__add_access_time(key)
                return
            large_dict.data_info[key] = DataInfo.IN_DISK
            large_dict.data.pop(key)
            large_dict.__remove_access_time(key)

    @staticmethod
    def delete_item(task):
        large_dict, key = task
        item_path = large_dict.get_key_storage_path(key)
        with large_dict.lock:
            if key not in large_dict.data_info:
                get_logger().warning("deleted key %s,ignore", key)
                return
            if large_dict.data_info[key] != DataInfo.PRE_DELETE:
                get_logger().warning("canceled key %s", key)
                return
            get_logger().warning("delete key %s", key)
            large_dict.data_info.pop(key)
            if key in large_dict.data:
                large_dict.data.pop(key)
            large_dict.__remove_access_time(key)
            if os.path.exists(item_path):
                shutil.rmtree(item_path)

    @staticmethod
    def read_item(task):
        large_dict, key = task

        if not large_dict.__change_data_info(
                key, DataInfo.PRE_LOAD, DataInfo.LOADING):
            return

        value = None
        try:
            value = torch.load(large_dict.get_key_storage_path(key))
        except Exception:
            get_logger().error("load key %s failed", key)
            with large_dict.lock:
                large_dict.data_info[key] = DataInfo.IN_DISK
                if large_dict.wait_fetch_event:
                    large_dict.fetch_event.set()

        with large_dict.lock:
            if key not in large_dict.data_info:
                get_logger().warning("canceled key %s", key)
                return
            if large_dict.data_info[key] != DataInfo.LOADING:
                get_logger().warning(
                    "canceled key %s, info is %s",
                    key,
                    large_dict.data_info[key])
                return
            large_dict.data[key] = value
            large_dict.data_info[key] = DataInfo.IN_MEMORY
            large_dict.__add_access_time(key)
            if large_dict.wait_fetch_event:
                large_dict.fetch_event.set()
                get_logger().debug("end fetch_event set")

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def set_in_memory_key_number(self, num):
        with self.lock:
            self.in_memory_key_number = num

    def set_permanent(self):
        self.permanent = True

    def flush_all(self):
        with self.lock:
            self.flush_all_once = True
            self.flush_thread.add_task(self)
        while True:
            time.sleep(1)
            with self.lock:
                if not self.flush_all_once:
                    return

    def print_data_info(self):
        cnt = {}
        with self.lock:
            for k, v in self.data_info.items():
                if v not in cnt:
                    cnt[v] = 1
                else:
                    cnt[v] += 1
        for k, v in cnt.items():
            get_logger().debug("%s => %s", k, v)

    def get_in_memeory_cnt(self):
        cnt = 0
        with self.lock:
            for v in self.data_info.values():
                if v in (DataInfo.IN_MEMORY, DataInfo.IN_MEMORY_NEW_DATA,):
                    cnt += 1
        return cnt

    def keys(self):
        real_keys = set()
        with self.lock:
            for k, v in self.data_info.items():
                if v != DataInfo.PRE_DELETE:
                    real_keys.add(k)
        return real_keys

    def prefetch(self, keys, ignore_unknown_keys=True):
        result = []
        for key in keys:
            with self.lock:
                if key not in self.data_info:
                    if ignore_unknown_keys:
                        continue
                    raise KeyError(key)
                if self.data_info[key] in (
                        DataInfo.PRE_LOAD, DataInfo.LOADING):
                    continue
                if key in self.data:
                    if self.data_info[key] != DataInfo.IN_MEMORY_NEW_DATA:
                        self.data_info[key] = DataInfo.IN_MEMORY
                    # self.__update_access_time(key)
                    result = [self.data[key]]
                    continue
                self.data_info[key] = DataInfo.PRE_LOAD
                self.fetch_queue.add_task((self, key))
        return result

    def release(self):
        get_logger().debug("release data structure")
        if self.permanent:
            self.flush_all()
        self.flush_thread.force_stop()
        self.fetch_queue.force_stop()
        self.delete_queue.force_stop()
        self.write_queue.force_stop()
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
        with self.lock:
            return key in self.data_info and self.data_info[key] != DataInfo.PRE_DELETE

    def __getitem__(self, key):
        get_logger().debug("before get lock")
        cur_time = time.time()
        result = self.prefetch([key])
        if result:
            get_logger().debug(
                "end get key %s %s", key, (time.time() - cur_time) * 1000
            )
            # get_logger().debug("end get key %s", key)
            return result[0]
        with self.lock:
            self.wait_fetch_event = True
        while self.fetch_event.wait():
            with self.lock:
                self.wait_fetch_event = False
                if self.data_info[key] in (
                    DataInfo.IN_MEMORY,
                    DataInfo.IN_MEMORY_NEW_DATA,
                ):
                    get_logger().debug("read key %s from disk", key)
                    return self.data[key]
                if self.data_info[key] == DataInfo.PRE_DELETE:
                    raise KeyError(key)
                self.fetch_event.clear()
                self.wait_fetch_event = True
        raise KeyError(key)

    def __setitem__(self, key, val):
        with self.lock:
            self.data[key] = val
            self.data_info[key] = DataInfo.IN_MEMORY_NEW_DATA
            # self.__update_access_time(key)

    def __delitem__(self, key):
        with self.lock:
            assert key in self.data_info
            self.data_info[key] = DataInfo.PRE_DELETE
            self.delete_queue.add_task((self, key))

    def __del__(self):
        self.release()

    def __change_data_info(self, key, from_info, to_info):
        with self.lock:
            if key not in self.data_info:
                get_logger().warning("no key %s,ignore", key)
                return False
            if self.data_info[key] != from_info:
                get_logger().warning(
                    "canceled key %s, info is %s", key, self.data_info[key]
                )
                return False
            self.data_info[key] = to_info
            return True

    def __remove_access_time(self, key):
        with self.lock:
            if key in self.LRU_keys:
                del self.LRU_keys[key]

    def __pop_expired_access_time(self):
        with self.lock:
            if len(self.LRU_keys) > self.in_memory_key_number or (
                self.flush_all_once and len(self.LRU_keys) > 0
            ):
                key = self.LRU_keys.popitem(last=False)[0]
                return True, key
            if not self.LRU_keys:
                self.flush_all_once = False
            return False, None

    def __update_access_time(self, key):
        get_logger().debug("begin update acc")
        with self.lock:
            self.LRU_keys.move_to_end(key)
        get_logger().debug("end update acc")

    def __add_access_time(self, key):
        get_logger().debug("begin add acc")
        with self.lock:
            self.LRU_keys[key] = None
            # assert len(self.data) == len(self.LRU_keys)
            if len(self.LRU_keys) > self.in_memory_key_number:
                self.flush_thread.add_task(self)
        get_logger().debug("end add acc")
