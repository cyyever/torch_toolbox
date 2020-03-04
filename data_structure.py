import multiprocessing

from enum import Enum, auto

import shutil
import os

import torch


class SentinelData:
    def __init__(self):
        self.flag = True


class DataInfo(Enum):
    IN_MEMORY = auto()
    IN_DISK = auto()
    IN_SAVING = auto()
    IN_LOADING = auto()


class LargeDict:
    def __init__(self, storage_dir=None):
        self.in_memory_key_number = 128
        self.data = dict()
        self.storage_dir = None
        self.manager = multiprocessing.Manager()
        self.data_info = self.manager.dict()
        self.lock = self.manager.lock()
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
        self.write_queue = multiprocessing.Queue()
        self.write_proc = multiprocessing.Process(
            target=LargeDict.write_item, args=(self.write_queue,)
        )
        self.write_proc.start()

    @staticmethod
    def write_item(q):
        while True:
            item = q.get()
            print("get item")
            if isinstance(item, SentinelData):
                print("exit process")
                break
            lock, data_info, key, value, stored_path = item
            with lock:
                if key not in data_info:
                    print("deleted key,ignore")
                    continue
                if data_info[key] == DataInfo.IN_SAVING:
                    torch.save(value, stored_path)
                    print("save item", stored_path)

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def set_in_memory_key_number(self, num):
        self.in_memory_key_number = num

    def save(self):
        self.permanent = True
        for k in self.in_memory_keys:
            self.__save_item(k)

    def keys(self):
        return self.data.keys()

    def clear(self):
        for k in self.keys():
            self.__delitem__(k)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        self.__flush_items(key)
        return self.__load_item(key)

    def __setitem__(self, key, val):
        self.__flush_items(key)
        assert val is not None
        self.data[key] = val
        self.in_memory_keys.add(key)

    def __delitem__(self, key):
        assert key in self.data
        self.data.remove(key)
        self.in_memory_keys.remove(key)
        print("remove item", self.__get_key_storage_path(key))
        shutil.rmtree(self.__get_key_storage_path(key))

    def __del__(self):
        print("del LargeDict")

        self.write_queue.put(SentinelData())
        print("end put ")
        self.write_queue.close()
        self.write_queue.join_thread()
        self.write_proc.join()
        print("end put join")
        if self.permanent:
            print("end LargeDict")
            return
        if self.storage_dir is not None:
            print("rmtree ", self.storage_dir)
            shutil.rmtree(self.storage_dir)
        print("end LargeDict")

    def __flush_items(self, excluded_key):
        while len(self.in_memory_keys) - 1 > self.in_memory_key_number:
            for other_key in list(self.in_memory_keys):
                if other_key == excluded_key:
                    continue
                self.__save_item(other_key)

    def __get_key_storage_path(self, key):
        if self.storage_dir is None:
            raise RuntimeError("no storage_dir")
        return os.path.join(self.storage_dir, str(key))

    def __save_item(self, key):
        assert key in self.data and self.data[key] is not None
        assert self.write_proc.is_alive()
        self.write_queue.put(
            (key, self.data[key], self.__get_key_storage_path(key)))
        self.data[key] = None
        self.in_memory_keys.remove(key)

    def __load_item(self, key):
        assert key in self.data
        print(self.data[key])

        if key not in self.in_memory_keys:
            value = torch.load(self.__get_key_storage_path(key))
            assert value is not None
            self.data[key] = value
            self.in_memory_keys.add(key)
        return self.data[key]
