import random
import copy
import pickle
import torch
import numpy

from cyy_naive_lib.log import get_logger


class ReproducibleEnv:
    def __init__(self):
        self.torch_seed = None
        self.randomlib_state = None
        self.numpy_state = None

    def __enter__(self):
        """
        https://pytorch.org/docs/stable/notes/randomness.html
        """
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(True)

        if self.torch_seed is not None:
            get_logger().warning("overwrite torch seed")
            assert isinstance(self.torch_seed, int)
            torch.manual_seed(self.torch_seed)
        else:
            get_logger().warning("collect torch seed")
            self.torch_seed = torch.initial_seed()
        if self.randomlib_state is not None:
            assert isinstance(self.randomlib_state, int)
            get_logger().warning("overwrite random lib state")
            random.setstate(self.randomlib_state)
        else:
            get_logger().warning("get random lib state")
            self.randomlib_state = random.getstate()

        if self.numpy_state is not None:
            get_logger().warning("overwrite numpy random lib state")
            numpy.random.setstate(copy.deepcopy(self.numpy_state))
        return self

    def __exit__(self, exc_type, exc_value, real_traceback):
        if real_traceback:
            return
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.set_deterministic(False)

    def save(self, path: str):
        with open(path, "wb") as f:
            return pickle.dump(
                {
                    "torch_seed": self.torch_seed,
                    "randomlib_state": self.randomlib_state,
                    "numpy_state": self.numpy_state,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            obj: dict = pickle.load(f)
            self.torch_seed = obj["torch_seed"]
            self.randomlib_state = obj["randomlib_state"]
            self.numpy_state = obj["numpy_state"]
