from multiprocessing import Process, Queue

import shutil
import os

import torch


class SentinelData:
    def __init__(self):
        self.flag = True


class LargeDict:
    def __init__(self, storage_dir=None):
        self.in_memory_key_number = 128
        self.in_memory_storage = dict()
        self.storage_dir = None
        if storage_dir is not None:
            self.set_storage_dir(storage_dir)
            self.in_memory_storage = {
                int(f): None
                for f in os.listdir(storage_dir)
                if os.path.isfile(os.path.join(storage_dir, f))
            }
            self.permanent = True
        else:
            self.permanent = False
        self.in_memory_storage_keys = set()
        self.write_queue = Queue()
        self.write_proc = Process(
            target=LargeDict.write_item, args=(
                self.write_queue,))
        self.write_proc.start()

    @staticmethod
    def write_item(q):
        while True:
            item = q.get()
            if isinstance(item, SentinelData):
                print("exit process")
                break
            value, stored_path = item
            torch.save(value, stored_path)

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def set_in_memory_key_number(self, num):
        self.in_memory_key_number = num

    def save(self):
        self.permanent = True
        for k in self.in_memory_storage_keys:
            self.__save_item(k)

    def keys(self):
        return self.in_memory_storage.keys()

    def __len__(self):
        return len(self.in_memory_storage)

    def __getitem__(self, key):
        self.__save_other_keys(key)
        return self.__load_item(key)

    def __setitem__(self, key, val):
        self.__save_other_keys(key)
        self.in_memory_storage[key] = val
        self.in_memory_storage_keys.add(key)

    def __delitem__(self, key):
        assert key in self.in_memory_storage
        self.in_memory_storage[key] = None
        self.in_memory_storage_keys.remove(key)
        shutil.rmtree(self.__get_key_storage_path(key))

    def __del__(self):
        self.write_queue.close()
        self.write_queue.join_thread()
        self.write_proc.join()
        if self.permanent:
            return
        if self.storage_dir is not None:
            print("rmtree ", self.storage_dir)
            shutil.rmtree(self.storage_dir)

    def __save_other_keys(self, excluded_key):
        while len(self.in_memory_storage_keys) - 1 > self.in_memory_key_number:
            for other_key in list(self.in_memory_storage_keys):
                if other_key == excluded_key:
                    continue
                self.__save_item(other_key)

    def __get_key_storage_path(self, key):
        if self.storage_dir is None:
            raise RuntimeError("no storage_dir")
        return os.path.join(self.storage_dir, str(key))

    def __save_item(self, key):
        assert key in self.in_memory_storage and self.in_memory_storage[key] is not None
        self.write_queue.put(
            (self.in_memory_storage[key], self.__get_key_storage_path(key))
        )
        self.in_memory_storage[key] = None
        self.in_memory_storage_keys.remove(key)

    def __load_item(self, key):
        assert key in self.in_memory_storage

        if key not in self.in_memory_storage_keys:
            value = torch.load(self.__get_key_storage_path(key))
            assert value is not None
            self.in_memory_storage[key] = value
            self.in_memory_storage_keys.add(key)
        return self.in_memory_storage[key]
