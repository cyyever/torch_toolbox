import uuid
import pickle
import os


class LargeDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_dir = None
        self.id = uuid.uuid4()
        self.in_memory_key_number = 128
        self.all_keys = set()

    def set_storage_dir(self, storage_dir):
        self.storage_dir = storage_dir

    def set_in_memory_key_number(self, num):
        self.in_memory_key_number = num

    def __getitem__(self, key):
        if key in self.all_keys:
            self.__load_item(key)
        return super().__getitem__(key)

    def __setitem__(self, key, val):
        self.all_keys.add(key)
        while super().__len__() > self.in_memory_key_number:
            for other_key in super().keys():
                if other_key == key:
                    continue
                self.__save_item(other_key)
                super().__delitem__(other_key)
        return super().__setitem__(self, key, val)

    def __delitem__(self, key):
        self.all_keys.remove(key)
        return super().__delitem__(key)

    def __get_key_storage_path(self, key):
        if self.storage_dir is None:
            raise RuntimeError("no storage_dir")
        return os.path.join(self.storage_dir, self.id + "_" + str(key))

    def __save_item(self, key):
        if not super().__contains__(key):
            raise RuntimeError("no key " + str(key))

        with open(os.path.join(self.__get_key_storage_path(key)), "wb") as f:
            pickle.dump(f, super().__getitem__(key))
        super().__delitem__(key)

    def __load_item(self, key):
        if key not in self.all_keys:
            raise RuntimeError("no key " + str(key))
        if super().__contains__(key):
            return

        with open(os.path.join(self.__get_key_storage_path(key)), "rb") as f:
            super().__setitem__(key, pickle.load(f))
        if not super().__contains__(key):
            raise RuntimeError("no key " + str(key) + " after load")
        return
