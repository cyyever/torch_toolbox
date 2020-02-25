import uuid
import shutil
import os

import torch


class LargeDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_dir = None
        self.id = str(uuid.uuid4())
        self.in_memory_key_number = 128
        self.all_keys = set()
        self.permanent = False

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir
        os.makedirs(os.path.join(self.storage_dir, self.id))

    def set_in_memory_key_number(self, num):
        self.in_memory_key_number = num

    def save(self):
        self.permanent = True
        for k in list(super().keys()):
            self.__save_item(k)

    def __getitem__(self, key):
        if key in self.all_keys:
            self.__load_item(key)
        return super().__getitem__(key)

    def __setitem__(self, key, val):
        self.all_keys.add(key)
        while super().__len__() > self.in_memory_key_number:
            for other_key in list(super().keys()):
                if other_key == key:
                    continue
                self.__save_item(other_key)
        return super().__setitem__(key, val)

    def __delitem__(self, key):
        self.all_keys.remove(key)
        return super().__delitem__(key)

    def __del__(self):
        if self.permanent:
            self.save()
            return
        if self.storage_dir is not None:
            shutil.rmtree(os.path.join(self.storage_dir, self.id))

    def __get_key_storage_path(self, key):
        if self.storage_dir is None:
            raise RuntimeError("no storage_dir")
        return os.path.join(self.storage_dir, self.id, str(key))

    def __save_item(self, key):
        if not super().__contains__(key):
            raise RuntimeError("no key " + str(key))

        torch.save(super().__getitem__(key), self.__get_key_storage_path(key))
        super().__delitem__(key)

    def __load_item(self, key):
        if key not in self.all_keys:
            raise RuntimeError("no key " + str(key))
        if super().__contains__(key):
            return

        super().__setitem__(torch.load(self.__get_key_storage_path(key)))
        if not super().__contains__(key):
            raise RuntimeError("no key " + str(key) + " after load")
        return
