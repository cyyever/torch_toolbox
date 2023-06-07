import os

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.reproducible_random_env import ReproducibleRandomEnv


class ReproducibleEnv(ReproducibleRandomEnv):
    def __init__(self) -> None:
        super().__init__()
        self.__torch_seed = None
        self.__torch_rng_state = None
        self.__torch_cuda_rng_state = None

    def enable(self) -> None:
        """
        https://pytorch.org/docs/stable/notes/randomness.html
        """
        with self.lock:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            torch.set_deterministic_debug_mode(2)

            if self.__torch_seed is not None:
                get_logger().debug("overwrite torch seed")
                torch.manual_seed(self.__torch_seed)
            else:
                get_logger().debug("collect torch seed")
                self.__torch_seed = torch.initial_seed()

            if self.__torch_cuda_rng_state is not None:
                get_logger().debug("overwrite torch cuda rng state")
                torch.cuda.set_rng_state_all(self.__torch_cuda_rng_state)
            elif torch.cuda.is_available():
                get_logger().debug("collect torch cuda rng state")
                self.__torch_cuda_rng_state = torch.cuda.get_rng_state_all()

            if self.__torch_rng_state is not None:
                get_logger().debug("overwrite torch rng state")
                torch.set_rng_state(self.__torch_rng_state)
            else:
                get_logger().debug("collect torch rng state")
                self.__torch_rng_state = torch.get_rng_state()
            super().enable()

    def disable(self) -> None:
        with ReproducibleEnv.lock:
            torch.use_deterministic_algorithms(False)
            torch.set_deterministic_debug_mode(0)
            super().disable()

    def get_state(self) -> dict:
        return super().get_state() | {
            "torch_seed": self.__torch_seed,
            "torch_cuda_rng_state": self.__torch_cuda_rng_state,
            "torch_rng_state": self.__torch_rng_state,
        }

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.__torch_seed = state["torch_seed"]
        self.__torch_cuda_rng_state = state["torch_cuda_rng_state"]
        self.__torch_rng_state = state["torch_rng_state"]


global_reproducible_env: ReproducibleEnv = ReproducibleEnv()
