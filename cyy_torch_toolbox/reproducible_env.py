import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import torch
from cyy_naive_lib.log import log_debug
from cyy_naive_lib.reproducible_random_env import ReproducibleRandomEnv


class ReproducibleEnv(ReproducibleRandomEnv):
    def __init__(self) -> None:
        super().__init__()
        self.__torch_seed: None | int = None
        self.__torch_rng_state: None | torch.Tensor = None
        self.__torch_accelerator_rng_state: list[torch.Tensor] | None = None

    @staticmethod
    def __get_accelerator_rng_state() -> list[torch.Tensor] | None:
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None:
            return None
        match accelerator.type:
            case "cuda":
                return torch.cuda.get_rng_state_all()
        return None

    @staticmethod
    def __set_accelerator_rng_state(state: list[torch.Tensor]) -> None:
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None:
            return
        match accelerator.type:
            case "cuda":
                torch.cuda.set_rng_state_all(state)

    @override
    def enable(self) -> None:
        """
        https://pytorch.org/docs/stable/notes/randomness.html
        """
        with self.lock:
            if torch.cuda.is_available():
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

            if (
                self.__torch_accelerator_rng_state is not None
                and torch.accelerator.is_available()
            ):
                log_debug("overwrite torch accelerator rng state")
                self.__set_accelerator_rng_state(self.__torch_accelerator_rng_state)
            elif torch.accelerator.is_available():
                log_debug("collect torch accelerator rng state")
                self.__torch_accelerator_rng_state = self.__get_accelerator_rng_state()

            if self.__torch_rng_state is not None:
                log_debug("overwrite torch rng state")
                torch.set_rng_state(self.__torch_rng_state)
            else:
                log_debug("collect torch rng state")
                self.__torch_rng_state = torch.get_rng_state()
            super().enable()

    @override
    def disable(self) -> None:
        with ReproducibleEnv.lock:
            torch.use_deterministic_algorithms(False)
            torch.set_deterministic_debug_mode(0)
            super().disable()

    @override
    def get_state(self) -> dict[str, Any]:
        return super().get_state() | {
            "torch_seed": self.__torch_seed,
            "torch_accelerator_rng_state": self.__torch_accelerator_rng_state,
            "torch_rng_state": self.__torch_rng_state,
        }

    @override
    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self.__torch_seed = state["torch_seed"]
        self.__torch_accelerator_rng_state = state["torch_accelerator_rng_state"]
        self.__torch_rng_state = state["torch_rng_state"]


global_reproducible_env: ReproducibleEnv = ReproducibleEnv()


@dataclass(kw_only=True)
class ReproducibleEnvConfig:
    make_reproducible_env: bool = True
    reproducible_env_load_path: Path | None = None

    def set_reproducible_env(self, save_dir: str | Path | None = None) -> None:
        if self.reproducible_env_load_path is not None:
            assert not global_reproducible_env.enabled
            global_reproducible_env.load(self.reproducible_env_load_path)
            self.make_reproducible_env = True

        if self.make_reproducible_env:
            global_reproducible_env.enable()
            if self.reproducible_env_load_path is None:
                assert save_dir is not None
                global_reproducible_env.save(save_dir)
