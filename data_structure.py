from enum import Enum, auto
import shutil
import os
import threading

import torch

from .task_queue import TaskQueue
from .log import get_logger


class DataInfo(Enum):
    IN_MEMORY = auto()
    IN_DISK = auto()
    PRE_SAVING = auto()
    SAVING = auto()
    PRE_LOAD = auto()
    LOADING = auto()
    PRE_DELETE = auto()


class LargeDict:
    def __init__(self, storage_dir=None):
        self.in_memory_key_number = 128
        self.lock = threading.lock()
        self.data = dict()
        self.data_info = dict()
        self.storage_dir = None

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
        self.fetch_queue = TaskQueue(LargeDict.read_item, 1)

    @staticmethod
    def write_item(task):
        large_dict, key = task
        value = None
        with large_dict.lock:
            if key not in large_dict.data_info:
                get_logger().info("deleted key", key, ",ignore")
                return
            if large_dict.data_info[key] != DataInfo.PRE_SAVING:
                get_logger().info("canceled key", key)
                return
            large_dict.data_info[key] = DataInfo.SAVING
            value = large_dict.data[key]

        item_path = large_dict.get_key_storage_path(key)
        torch.save(value, item_path)
        with large_dict.lock:
            if key not in large_dict.data_info:
                shutil.rmtree(item_path)
                return
            if large_dict.data_info[key] == DataInfo.SAVING:
                large_dict.data_info[key] = DataInfo.IN_DISK
                large_dict.data.remove(key)

    @staticmethod
    def delete_item(task):
        large_dict, key = task
        item_path = large_dict.get_key_storage_path(key)
        with large_dict.lock:
            if key not in large_dict.data_info:
                get_logger().info("deleted key", key, ",ignore")
                return
            if large_dict.data_info[key] != DataInfo.PRE_DELETE:
                get_logger().info("canceled key", key)
                return
            large_dict.data_info.remove(key)
            large_dict.data.remove(key)
            shutil.rmtree(item_path)

    @staticmethod
    def read_item(task):
        large_dict, key = task
        with large_dict.lock:
            if key not in large_dict.data_info:
                get_logger().info("deleted key", key, ",ignore")
                return
            if large_dict.data_info[key] != DataInfo.PRE_LOAD:
                get_logger().info("canceled key", key)
                return
            large_dict.data_info[key] = DataInfo.LOADING

        value = torch.load(large_dict.get_key_storage_path(key))
        with large_dict.lock:
            if key not in large_dict.data_info:
                return
            if large_dict.data_info[key] == DataInfo.LOADING:
                large_dict.data_info[key] = DataInfo.IN_MEMORY
                large_dict.data[key] = value

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def set_in_memory_key_number(self, num):
        self.in_memory_key_number = num

    def save(self):
        self.permanent = True
        for k in self.keys():
            self.__save_item(k)

    def keys(self):
        real_keys = set()
        with self.lock:
            for k, v in self.data_info.items():
                if v != DataInfo.PRE_DELETE:
                    real_keys.add(k)
        return real_keys

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, key):
        self.__flush_items(key)
        return self.__load_item(key)

    def __setitem__(self, key, val):
        self.__flush_items(key)
        with self.lock:
            self.data_info[key] = DataInfo.IN_MEMORY
            self.data[key] = val

    def __delitem__(self, key):
        with self.lock:
            assert key in self.data
            assert key in self.data_info
            self.data_info[key] = DataInfo.PRE_DELETE
            self.delete_queue.add_task((self, key))

    def __del__(self):
        print("del LargeDict")
        self.fetch_queue.stop()
        self.delete_queue.stop()
        self.write_queue.stop()
        print("end put join")
        if self.permanent:
            print("end LargeDict")
            return
        if self.storage_dir is not None:
            print("rmtree ", self.storage_dir)
            shutil.rmtree(self.storage_dir)
        print("end LargeDict")

    def __flush_items(self, excluded_key):
        real_keys = self.keys()
        while len(real_keys) - 1 > self.in_memory_key_number:
            for other_key in real_keys:
                if other_key == excluded_key:
                    continue
                self.__save_item(other_key)

    def get_key_storage_path(self, key):
        if self.storage_dir is None:
            raise RuntimeError("no storage_dir")
        return os.path.join(self.storage_dir, str(key))

    def __save_item(self, key):
        with self.lock:
            if key not in self.data_info:
                return
            if self.data_info[key] == DataInfo.IN_MEMORY:
                self.data_info[key] = DataInfo.PRE_SAVING
                self.write_queue.add_task((self, key))

    def __load_item(self, key):
        with self.lock:
            if key not in self.data_info:
                raise KeyError(key)
            if key in self.data:
                self.data_info[key] = DataInfo.IN_MEMORY
                return self.data[key]
            LargeDict.read_item((self, key))
            if key not in self.data:
                raise KeyError(key)
            return self.data[key]
