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


class LargeDict:
    def __init__(self, storage_dir=None):
        self.__in_memory_key_number = 128
        self.lock = threading.RLock()
        self.data = dict()
        self.data_info = dict()
        self.__LRU_keys = None
        self.__create_LRU_keys()
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
        # note: flush_thread number must be 1
        self.flush_thread = TaskQueue(LargeDict.flush_item, 1)
        self.flush_all_once = False
        self.fetch_event = threading.Event()
        self.wait_fetch_event = False
        self.in_flush = False

    @staticmethod
    def flush_item(task):
        large_dict = task
        LRU_keys = collections.OrderedDict()
        while True:
            LRU_keys_len = large_dict.__get_LRU_key_num()
            if LRU_keys_len == 0:
                break

            LRU_keys_list = [None] * LRU_keys_len

            with large_dict.lock:
                for i in range(LRU_keys_len):
                    LRU_keys_list[i] = large_dict.__LRU_keys[i]
                large_dict.__next_LRU_key_index = 0
            for i in range(LRU_keys_len):
                k = LRU_keys_list[i]
                if k in LRU_keys:
                    LRU_keys.move_to_end(k)
                else:
                    LRU_keys[k] = None
            LRU_keys_len = len(LRU_keys)
            if LRU_keys_len == 0:
                break

            threshold = None
            with large_dict.lock:
                threshold = large_dict.__get_in_memory_key_number()

            key = None
            if LRU_keys_len > threshold:
                key = LRU_keys.popitem(last=False)[0]
            else:
                break

            with large_dict.lock:
                data_info = large_dict.data_info.get(key, None)
                assert data_info in (
                    DataInfo.IN_MEMORY,
                    DataInfo.IN_MEMORY_NEW_DATA)
                if data_info is None:
                    continue
                if data_info == DataInfo.IN_MEMORY_NEW_DATA:
                    large_dict.data_info[key] = DataInfo.PRE_SAVING
                    large_dict.write_queue.add_task((large_dict, key))
                else:
                    large_dict.data_info[key] = DataInfo.IN_DISK
                    large_dict.data.pop(key)

        LRU_keys_list.clear()
        while LRU_keys:
            key = LRU_keys.popitem(last=False)[0]
            LRU_keys_list.append(key)

        continue_flush = False
        LRU_keys_len = len(LRU_keys_list)
        with large_dict.lock:
            large_dict.in_flush = False
            if LRU_keys_len != 0:
                large_dict.__LRU_keys = LRU_keys_list + large_dict.__LRU_keys
                large_dict.__next_LRU_key_index += LRU_keys_len
            if large_dict.flush_all_once:
                if not large_dict.__LRU_keys:
                    large_dict.flush_all_once = False
                else:
                    continue_flush = True
        if continue_flush:
            large_dict.flush_thread.add_task(large_dict)

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
            data_info = large_dict.data_info.get(key, None)
            if data_info is None:
                if os.path.exists(item_path):
                    shutil.rmtree(item_path)
                return
            if data_info != DataInfo.SAVING:
                get_logger().warning("canceled key %s, info is %s", key, data_info)
                return
            if not succ_flag:
                large_dict.data_info[key] = DataInfo.IN_MEMORY_NEW_DATA
                large_dict.__update_item_access_time(key)
                return
            large_dict.data_info[key] = DataInfo.IN_DISK
            large_dict.data.pop(key)

    @staticmethod
    def delete_item(task):
        large_dict, key = task
        item_path = large_dict.get_key_storage_path(key)
        with large_dict.lock:
            data_info = large_dict.data_info.get(key, None)
            if data_info is not None:
                get_logger().warning("canceled key delete %s", key)
                return
            get_logger().warning("delete key %s", key)
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
            if not large_dict.__change_data_info(
                key, DataInfo.LOADING, DataInfo.IN_MEMORY
            ):
                return
            large_dict.data[key] = value
            large_dict.__update_item_access_time(key)
            if large_dict.wait_fetch_event:
                large_dict.fetch_event.set()
                get_logger().debug("end fetch_event set")

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def set_in_memory_key_number(self, num):
        with self.lock:
            self.__in_memory_key_number = num

    def set_permanent(self):
        self.permanent = True

    def flush_all(self):
        with self.lock:
            self.flush_all_once = True
            self.flush_thread.add_task(self)

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
        with self.lock:
            return self.data_info.keys()

    def prefetch(self, keys, ignore_unknown_keys=True):
        result = []
        for key in keys:
            with self.lock:
                data_info = self.data_info.get(key, None)
                if data_info is None:
                    if ignore_unknown_keys:
                        continue
                    raise KeyError(key)
                if data_info in (DataInfo.PRE_LOAD, DataInfo.LOADING):
                    continue
                if key in self.data:
                    if data_info != DataInfo.IN_MEMORY_NEW_DATA:
                        self.data_info[key] = DataInfo.IN_MEMORY

                    self.__update_item_access_time(key)
                    result = [self.data[key]]
                    continue
                self.data_info[key] = DataInfo.PRE_LOAD
                self.fetch_queue.add_task((self, key))
        return result

    def release(self):
        get_logger().info("release data structure")
        if self.permanent:
            self.flush_all()
        self.fetch_queue.force_stop()
        self.flush_thread.stop()
        self.delete_queue.stop()
        self.write_queue.stop()
        self.data = None
        self.in_flush = False

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
            return key in self.data_info

    def __getitem__(self, key):
        get_logger().debug("before get lock")
        cur_time = time.time()
        result = self.prefetch([key])
        if result:
            get_logger().warning(
                "end get key %s %s", key, (time.time() - cur_time) * 1000
            )
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
                    get_logger().warning("read key %s from disk", key)
                    return self.data[key]
                self.fetch_event.clear()
                self.wait_fetch_event = True
        raise KeyError(key)

    def __setitem__(self, key, val):
        cur_time = time.time()
        with self.lock:
            self.data[key] = val
            self.data_info[key] = DataInfo.IN_MEMORY_NEW_DATA
            get_logger().warning(
                "set data ,use time %s", (time.time() - cur_time) * 1000
            )
            cur_time = time.time()
            self.__update_item_access_time(key)
            get_logger().warning(
                "update access time ,use time %s",
                (time.time() - cur_time) * 1000)

    def __delitem__(self, key):
        with self.lock:
            data_info = self.data_info.pop(key, None)
            if data_info is None:
                raise KeyError(key)
            self.data.pop(key, None)
            self.delete_queue.add_task((self, key))

    def __del__(self):
        self.release()

    def __get_in_memory_key_number(self):
        if self.flush_all_once:
            return 0
        return self.__in_memory_key_number

    def __change_data_info(self, key, from_info, to_info):
        with self.lock:
            data_info = self.data_info.pop(key, None)
            if data_info is None:
                get_logger().warning("no key %s,ignore", key)
                return False
            if data_info != from_info:
                get_logger().warning("canceled key %s, info is %s", key, data_info)
                return False
            self.data_info[key] = to_info
            return True

    def __create_LRU_keys(self):
        if self.__LRU_keys is None:
            self.__LRU_keys = [None] * (self.__in_memory_key_number * 2)
        self.__LRU_key_len = len(self.__LRU_keys)
        self.__next_LRU_key_index = 0

    def __get_LRU_key_num(self):
        with self.lock:
            return self.__next_LRU_key_index

    def __update_item_access_time(self, key):
        cur_time = time.time()
        if self.__next_LRU_key_index < self.__LRU_key_len:
            self.__LRU_keys[self.__next_LRU_key_index] = key
            get_logger().warning(
                "end inplace %s ", (time.time() - cur_time) * 1000,
            )
        else:
            self.__LRU_key_len += 1
            self.__LRU_keys.append(key)
            get_logger().warning(
                "end  append %s len %s ",
                (time.time() - cur_time) * 1000,
                len(self.__LRU_keys),
            )
        self.__next_LRU_key_index += 1
        cur_time = time.time()
        if (not self.in_flush and self.__next_LRU_key_index >
                self.__get_in_memory_key_number() * 1.5):
            get_logger().warning(
                "end check in flush %s ", (time.time() - cur_time) * 1000
            )
            cur_time = time.time()
            self.flush_thread.add_task(self)
            get_logger().warning("end add_task %s ", (time.time() - cur_time) * 1000)
            self.in_flush = True
