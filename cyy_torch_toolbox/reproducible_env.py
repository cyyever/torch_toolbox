import copy
import os
import pickle
import random
import threading

import numpy
import torch
from cyy_naive_lib.log import get_logger


class ReproducibleEnv:
    lock = threading.RLock()

    def __init__(self):
        self.__torch_seed = None
        self.__randomlib_state = None
        self.__numpy_state = None
        self.__enabled = False
        self.__last_seed_path = None

    @property
    def enabled(self):
        return self.__enabled

    @property
    def last_seed_path(self):
        return self.__last_seed_path

    def enable(self):
        """
        https://pytorch.org/docs/stable/notes/randomness.html
        """
        with ReproducibleEnv.lock:
            if self.__enabled:
                get_logger().warning("use reproducible env")
            else:
                get_logger().warning("initialize and use reproducible env")

            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.set_deterministic(True)
            torch.use_deterministic_algorithms(True)

            if self.__torch_seed is not None:
                get_logger().warning("overwrite torch seed")
                assert isinstance(self.__torch_seed, int)
                torch.manual_seed(self.__torch_seed)
            else:
                get_logger().warning("collect torch seed")
                self.__torch_seed = torch.initial_seed()
            assert self.__torch_seed is not None

            if self.__randomlib_state is not None:
                get_logger().warning("overwrite random lib state")
                random.setstate(self.__randomlib_state)
            else:
                get_logger().warning("get random lib state")
                self.__randomlib_state = random.getstate()
            assert self.__randomlib_state is not None

            if self.__numpy_state is not None:
                get_logger().warning("overwrite numpy random lib state")
                numpy.random.set_state(copy.deepcopy(self.__numpy_state))
            else:
                get_logger().warning("get numpy random lib state")
                self.__numpy_state = numpy.random.get_state()
            assert self.__numpy_state is not None
            self.__enabled = True

    def disable(self):
        with ReproducibleEnv.lock:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG")
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.set_deterministic(False)
            torch.use_deterministic_algorithms(False)

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_value, real_traceback):
        if real_traceback:
            return
        self.disable()

    def save(self, save_dir: str):
        with ReproducibleEnv.lock:
            assert self.__enabled
            os.makedirs(save_dir, exist_ok=True)
            seed_path = os.path.join(save_dir, "random_seed")
            get_logger().warning("save reproducible env to %s", seed_path)
            self.__last_seed_path = seed_path
            with open(seed_path, "wb") as f:
                return pickle.dump(
                    {
                        "torch_seed": self.__torch_seed,
                        "randomlib_state": self.__randomlib_state,
                        "numpy_state": self.__numpy_state,
                    },
                    f,
                )

    def load(self, path: str):
        with ReproducibleEnv.lock:
            assert not self.__enabled
            with open(path, "rb") as f:
                get_logger().warning("load reproducible env from %s", path)
                obj: dict = pickle.load(f)
                self.__torch_seed = obj["torch_seed"]
                self.__randomlib_state = obj["randomlib_state"]
                self.__numpy_state = obj["numpy_state"]
                self.__enabled = False

    def load_last_seed(self):
        self.load(self.last_seed_path)


global_reproducible_env: ReproducibleEnv = ReproducibleEnv()
