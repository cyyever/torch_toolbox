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
        self.__torch_rng_state = None
        self.__torch_cuda_rng_state = None
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
                get_logger().warning("%s use reproducible env", id(self))
            else:
                get_logger().warning("%s initialize and use reproducible env", id(self))

            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            try:
                torch.set_deterministic_debug_mode(2)
            except BaseException:
                pass

            if self.__torch_seed is not None:
                get_logger().debug("overwrite torch seed")
                torch.manual_seed(self.__torch_seed)
            else:
                get_logger().debug("collect torch seed")
                self.__torch_seed = torch.initial_seed()

            if self.__torch_cuda_rng_state is not None:
                get_logger().debug("overwrite torch cuda rng state")
                torch.cuda.set_rng_state_all(self.__torch_cuda_rng_state)
            else:
                get_logger().debug("collect torch cuda rng state")
                self.__torch_cuda_rng_state = torch.cuda.get_rng_state_all()

            if self.__torch_rng_state is not None:
                get_logger().debug("overwrite torch cuda rng state")
                torch.set_rng_state(self.__torch_rng_state)
            else:
                get_logger().debug("collect torch rng state")
                self.__torch_rng_state = torch.get_rng_state()

            if self.__randomlib_state is not None:
                get_logger().debug("overwrite random lib state")
                random.setstate(self.__randomlib_state)
            else:
                get_logger().debug("get random lib state")
                self.__randomlib_state = random.getstate()

            if self.__numpy_state is not None:
                get_logger().debug("overwrite numpy random lib state")
                numpy.random.set_state(copy.deepcopy(self.__numpy_state))
            else:
                get_logger().debug("get numpy random lib state")
                self.__numpy_state = numpy.random.get_state()
            self.__enabled = True

    def disable(self):
        get_logger().warning("disable reproducible env")
        with ReproducibleEnv.lock:
            torch.use_deterministic_algorithms(False)
            try:
                torch.set_deterministic_debug_mode(0)
            except BaseException:
                pass
            self.__enabled = False

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_value, real_traceback):
        if real_traceback:
            return
        self.disable()

    def save(self, save_dir: str):
        seed_path = os.path.join(save_dir, "random_seed")
        get_logger().warning("%s save reproducible env to %s", id(self), seed_path)
        with ReproducibleEnv.lock:
            assert self.__enabled
            os.makedirs(save_dir, exist_ok=True)
            self.__last_seed_path = seed_path
            with open(seed_path, "wb") as f:
                return pickle.dump(
                    {
                        "torch_seed": self.__torch_seed,
                        "torch_cuda_rng_state": self.__torch_cuda_rng_state,
                        "torch_rng_state": self.__torch_rng_state,
                        "randomlib_state": self.__randomlib_state,
                        "numpy_state": self.__numpy_state,
                    },
                    f,
                )

    def load(self, path: str):
        with ReproducibleEnv.lock:
            assert not self.__enabled
            with open(path, "rb") as f:
                get_logger().warning("%s load reproducible env from %s", id(self), path)
                obj: dict = pickle.load(f)
                self.__torch_seed = obj["torch_seed"]
                self.__torch_cuda_rng_state = obj["torch_cuda_rng_state"]
                self.__randomlib_state = obj["randomlib_state"]
                self.__numpy_state = obj["numpy_state"]

    def load_last_seed(self):
        self.load(self.last_seed_path)


global_reproducible_env: ReproducibleEnv = ReproducibleEnv()
