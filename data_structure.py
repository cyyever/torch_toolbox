from enum import Enum, auto
import shutil
import os
import time
import threading

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
        self.time_to_key = []
        self.storage_dir = None
        self.flush_all_once = False
        self.fetch_event = threading.Event()

        if storage_dir is not None:
            self.set_storage_dir(storage_dir)
            self.data_info = {
                int(f): DataInfo.IN_DISK
                for f in os.listdir(storage_dir)
                if os.path.isfile(os.path.join(storage_dir, f))
            }
            self.permanent = True
        else:
            self.permanent = False

        self.write_queue = TaskQueue(LargeDict.write_item, 1)
        self.delete_queue = TaskQueue(LargeDict.delete_item, 1)
        self.fetch_queue = TaskQueue(LargeDict.read_item, 2)
        self.flush_thread = TaskQueue(LargeDict.flush_item, 1)

    @staticmethod
    def flush_item(task):
        large_dict = task
        with large_dict.lock:
            if len(large_dict.time_to_key) == 0:
                large_dict.flush_all_once = False
                return

            while len(large_dict.time_to_key) > large_dict.in_memory_key_number or (
                    large_dict.flush_all_once and len(large_dict.time_to_key) > 0):
                key = large_dict.time_to_key.pop(0)
                if large_dict.data_info[key] == DataInfo.IN_MEMORY_NEW_DATA:
                    large_dict.data_info[key] = DataInfo.PRE_SAVING
                    large_dict.write_queue.add_task((large_dict, key))

    @staticmethod
    def write_item(task):
        large_dict, key = task
        value = None
        with large_dict.lock:
            if key not in large_dict.data_info:
                get_logger().warning("deleted key %s,ignore", key)
                return
            if large_dict.data_info[key] != DataInfo.PRE_SAVING:
                get_logger().warning(
                    "canceled key %s, info is %s",
                    key,
                    large_dict.data_info[key])
                return
            large_dict.data_info[key] = DataInfo.SAVING
            value = large_dict.data[key]

        item_path = large_dict.get_key_storage_path(key)
        save_flag = True
        try:
            torch.save(value, item_path)
        except Exception:
            get_logger().error("save key %s failed,keep it in memory", key)
            save_flag = False

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
            if not save_flag:
                large_dict.data_info[key] = DataInfo.IN_MEMORY_NEW_DATA
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
        with large_dict.lock:
            if key not in large_dict.data_info:
                get_logger().warning("deleted key %s,ignore", key)
                return
            if large_dict.data_info[key] != DataInfo.PRE_LOAD:
                get_logger().warning(
                    "canceled key %s, info is %s",
                    key,
                    large_dict.data_info[key])
                return
            large_dict.data_info[key] = DataInfo.LOADING

        value = None
        try:
            value = torch.load(large_dict.get_key_storage_path(key))
        except Exception:
            get_logger().error("load key %s failed,delete it", key)
            large_dict.__delitem__(key)
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
            large_dict.__update_access_time(key)
            large_dict.fetch_event.set()

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def set_in_memory_key_number(self, num):
        with self.lock:
            self.in_memory_key_number = num

    def flush_all(self):
        self.permanent = True
        with self.lock:
            self.flush_all_once = True
            self.flush_thread.add_task(self)
        while True:
            time.sleep(1)
            with self.lock:
                if len(self.time_to_key) == 0:
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
                    self.__update_access_time(key)
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
        result = self.prefetch([key])
        if result:
            return result[0]
        while self.fetch_event.wait():
            with self.lock:
                if self.data_info[key] in (
                    DataInfo.IN_MEMORY,
                    DataInfo.IN_MEMORY_NEW_DATA,
                ):
                    get_logger().debug("read key %s from disk", key)
                    return self.data[key]
                if self.data_info[key] == DataInfo.PRE_DELETE:
                    raise KeyError(key)
                self.fetch_event.clear()
        raise KeyError(key)

    def __setitem__(self, key, val):
        with self.lock:
            self.data[key] = val
            self.data_info[key] = DataInfo.IN_MEMORY_NEW_DATA
            self.__update_access_time(key)

    def __delitem__(self, key):
        with self.lock:
            assert key in self.data_info
            self.data_info[key] = DataInfo.PRE_DELETE
            self.delete_queue.add_task((self, key))

    def __del__(self):
        self.release()

    def __remove_access_time(self, key):
        with self.lock:
            try:
                self.time_to_key.remove(key)
            except ValueError:
                pass

    def __update_access_time(self, key):
        with self.lock:
            self.__remove_access_time(key)
            self.time_to_key.append(key)
            if len(self.time_to_key) > self.in_memory_key_number:
                self.flush_thread.add_task(self)
