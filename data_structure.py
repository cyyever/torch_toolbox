import shutil
import os

import torch


class LargeDict(dict):
    def __init__(self, storage_dir=None):
        super().__init__()
        self.in_memory_key_number = 128
        self.key_sync = dict()
        self.storage_dir = None
        if storage_dir is not None:
            self.set_storage_dir(storage_dir)
            self.key_sync = {
                int(f): True
                for f in os.listdir(storage_dir)
                if os.path.isfile(os.path.join(storage_dir, f))
            }
            self.permanent = True
        else:
            self.permanent = False

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def set_in_memory_key_number(self, num):
        self.in_memory_key_number = num

    def save(self):
        self.permanent = True
        for k in list(super().keys()):
            self.__save_item(k)

    def __getitem__(self, key):
        self.__save_other_keys(key)
        return self.__load_item(key)

    def __setitem__(self, key, val):
        self.__save_other_keys(key)
        res = super().__setitem__(key, val)
        self.key_sync[key] = True
        return res

    def __delitem__(self, key):
        self.key_sync.remove(key)
        return super().__delitem__(key)

    def __del__(self):
        if self.permanent:
            self.save()
            return
        if self.storage_dir is not None:
            shutil.rmtree(self.storage_dir)

    def __save_other_keys(self, excluded_key):
        while super().__len__() - 1 > self.in_memory_key_number:
            for other_key in list(super().keys()):
                if other_key == excluded_key:
                    continue
                self.__save_item(other_key)

    def __get_key_storage_path(self, key):
        if self.storage_dir is None:
            raise RuntimeError("no storage_dir")
        return os.path.join(self.storage_dir, str(key))

    def __save_item(self, key):
        if not self.key_sync[key]:
            torch.save(
                super().__getitem__(key),
                self.__get_key_storage_path(key))
            self.key_sync[key] = True
        super().__delitem__(key)

    def __load_item(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        value = torch.load(self.__get_key_storage_path(key))
        super().__setitem__(key, value)
        self.key_sync[key] = True
        return value
