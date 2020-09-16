import random
import torch

from cyy_naive_lib.log import get_logger


class ReproducibleEnv:
    def __init__(self):
        self.torch_seed = None
        self.randomlib_state = None

    def __enter__(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        return self

    def __exit__(self, exc_type, exc_value, real_traceback):
        if real_traceback:
            return
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
