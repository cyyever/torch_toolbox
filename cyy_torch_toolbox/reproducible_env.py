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
        self.torch_seed = None
        self.randomlib_state = None
        self.numpy_state = None
        self.enabled = False

    def enable(self):
        """
        https://pytorch.org/docs/stable/notes/randomness.html
        """
        with ReproducibleEnv.lock:
            if self.enabled:
                get_logger().warning("use reproducible env")
            else:
                get_logger().warning("initialize and use reproducible env")

            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.set_deterministic(True)
            torch.use_deterministic_algorithms(True)

            if self.torch_seed is not None:
                get_logger().warning("overwrite torch seed")
                assert isinstance(self.torch_seed, int)
                torch.manual_seed(self.torch_seed)
            else:
                get_logger().warning("collect torch seed")
                self.torch_seed = torch.initial_seed()
            assert self.torch_seed is not None

            if self.randomlib_state is not None:
                get_logger().warning("overwrite random lib state")
                random.setstate(self.randomlib_state)
            else:
                get_logger().warning("get random lib state")
                self.randomlib_state = random.getstate()
            assert self.randomlib_state is not None

            if self.numpy_state is not None:
                get_logger().warning("overwrite numpy random lib state")
                numpy.random.set_state(copy.deepcopy(self.numpy_state))
            else:
                get_logger().warning("get numpy random lib state")
                self.numpy_state = numpy.random.get_state()
            assert self.numpy_state is not None
            self.enabled = True

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
            assert self.enabled
            os.makedirs(save_dir, exist_ok=True)
            env_path = os.path.join(save_dir, "reproducible_env")
            get_logger().warning("save reproducible env to %s", env_path)
            with open(env_path, "wb") as f:
                return pickle.dump(
                    {
                        "torch_seed": self.torch_seed,
                        "randomlib_state": self.randomlib_state,
                        "numpy_state": self.numpy_state,
                    },
                    f,
                )

    def load(self, path: str):
        with ReproducibleEnv.lock:
            assert not self.enabled
            with open(path, "rb") as f:
                get_logger().warning("load reproducible env from %s", path)
                obj: dict = pickle.load(f)
                self.torch_seed = obj["torch_seed"]
                self.randomlib_state = obj["randomlib_state"]
                self.numpy_state = obj["numpy_state"]
                self.enabled = False


global_reproducible_env: ReproducibleEnv = ReproducibleEnv()
