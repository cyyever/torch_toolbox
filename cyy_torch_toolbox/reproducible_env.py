import os
from dataclasses import dataclass

import torch
from cyy_naive_lib.log import log_debug
from cyy_naive_lib.reproducible_random_env import ReproducibleRandomEnv


class ReproducibleEnv(ReproducibleRandomEnv):
    def __init__(self) -> None:
        super().__init__()
        self.__torch_seed: None | int = None
        self.__torch_rng_state: None | torch.Tensor = None
        self.__torch_cuda_rng_state: None | list = None

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
                log_debug("overwrite torch seed")
                torch.manual_seed(self.__torch_seed)
            else:
                log_debug("collect torch seed")
                self.__torch_seed = torch.initial_seed()

            if self.__torch_cuda_rng_state is not None:
                log_debug("overwrite torch cuda rng state")
                torch.cuda.set_rng_state_all(self.__torch_cuda_rng_state)
            elif torch.cuda.is_available():
                log_debug("collect torch cuda rng state")
                self.__torch_cuda_rng_state = torch.cuda.get_rng_state_all()

            if self.__torch_rng_state is not None:
                log_debug("overwrite torch rng state")
                torch.set_rng_state(self.__torch_rng_state)
            else:
                log_debug("collect torch rng state")
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


@dataclass(kw_only=True)
class ReproducibleEnvConfig:
    make_reproducible_env: bool = True
    reproducible_env_load_path: str | None = None

    def set_reproducible_env(self, save_dir: str | None = None) -> None:
        if self.reproducible_env_load_path is not None:
            assert not global_reproducible_env.enabled
            global_reproducible_env.load(self.reproducible_env_load_path)
            self.make_reproducible_env = True

        if self.make_reproducible_env:
            global_reproducible_env.enable()
            if self.reproducible_env_load_path is None:
                assert save_dir is not None
                global_reproducible_env.save(save_dir)
