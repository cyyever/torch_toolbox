import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from cyy_naive_lib.log import log_info

from .metric import Metric

_LINEAR_CLASSES: tuple[type, ...] = (nn.Linear,)
try:
    import bitsandbytes as bnb

    _LINEAR_CLASSES = (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
except ImportError:
    pass


class FLOPMetric(Metric):
    """Profiles training FLOP count and FLOPS throughput via module forward hooks.

    Registers forward hooks on all Linear-like modules (including bitsandbytes
    quantized layers) to count FLOPs as 2 * in_features * out_features * tokens.
    This is kernel-agnostic and works with bnb::matmul_4bit, aten::mm, etc.

    Reports:
        total_flop  — total floating point operations (hardware-independent)
        flops       — throughput in FLOP/s using wall clock time

    As executor metric hook:
        Automatically managed by PerformanceMetric / HookConfig.

    With TRL SFTTrainer:
        metric = FLOPMetric(model=model)
        trainer = SFTTrainer(..., callbacks=[metric.as_trainer_callback()])
        trainer.train()
        print(metric.get_total_flop(), metric.get_flops())
    """

    def __init__(
        self,
        model: torch.nn.Module,
        save_dir: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._save_dir = Path(save_dir) if save_dir is not None else None
        self._hooks: list = []
        self._total_flop: float = 0
        self._bnb_flop: float = 0
        self._flops: float = 0
        self._start_time: float = 0
        self._profiled: bool = False

    def _forward_hook(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        _output: Any,
    ) -> None:
        inp = input[0]
        tokens = inp.shape[:-1].numel()
        in_features: int = module.in_features  # type: ignore[assignment]
        out_features: int = module.out_features  # type: ignore[assignment]
        flop = 2.0 * in_features * out_features * tokens
        self._total_flop += flop
        if isinstance(module, _LINEAR_CLASSES[1:]):
            self._bnb_flop += flop

    def start(self) -> None:
        self._total_flop = 0
        self._bnb_flop = 0
        self._flops = 0
        self._start_time = time.monotonic()
        for module in self._model.modules():
            if isinstance(module, _LINEAR_CLASSES):
                self._hooks.append(
                    module.register_forward_hook(self._forward_hook)
                )

    def step(self) -> None:
        pass

    def stop(self) -> None:
        elapsed = time.monotonic() - self._start_time
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if self._total_flop > 0:
            self._flops = self._total_flop / elapsed if elapsed > 0 else 0
        log_info(
            "total_flop: %.4e (bnb: %.4e, %.1f%%), wall_time: %.2fs, flops: %.4e",
            self._total_flop,
            self._bnb_flop,
            self._bnb_flop / self._total_flop * 100 if self._total_flop > 0 else 0,
            elapsed,
            self._flops,
        )
        self._profiled = True
        self._save()

    def _save(self) -> None:
        if self._save_dir is None:
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        path = self._save_dir / "flop_data.json"
        data: dict = {}
        if path.is_file():
            with open(path, encoding="utf8") as f:
                data = json.load(f)
        entry = {"total_flop": self._total_flop, "bnb_flop": self._bnb_flop, "flops": self._flops}
        data.setdefault("rounds", []).append(entry)
        with open(path, "w", encoding="utf8") as f:
            json.dump(data, f, indent=2)
        log_info("FLOP data saved to %s", path)

    def get_total_flop(self) -> float:
        return self._total_flop

    def get_flops(self) -> float:
        return self._flops

    def as_trainer_callback(self) -> Any:
        """Returns a TrainerCallback for use with TRL SFTTrainer / HF Trainer."""
        from transformers import TrainerCallback

        metric = self

        class _Callback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                metric.start()

            def on_step_end(self, args, state, control, **kwargs):
                metric.step()

            def on_train_end(self, args, state, control, **kwargs):
                metric.stop()

        return _Callback()

    def _before_epoch(self, **kwargs: Any) -> None:
        if not self._profiled:
            self.start()

    def _after_batch(self, **kwargs: Any) -> None:
        self.step()

    def _after_epoch(self, epoch: int, **kwargs: Any) -> None:
        if not self._profiled and self._hooks:
            self.stop()
            self._set_epoch_metric(epoch, "total_flop", self._total_flop)
            self._set_epoch_metric(epoch, "flops", self._flops)
