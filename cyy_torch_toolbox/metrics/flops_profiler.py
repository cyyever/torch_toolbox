import json
import time
from pathlib import Path
from typing import Any

import torch
from cyy_naive_lib.log import log_info, log_warning
from torch.profiler import ProfilerActivity

from .metric import Metric


class FLOPMetric(Metric):
    """Profiles training FLOP count and FLOPS throughput using torch.profiler.

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
        self._prof: torch.profiler.profile | None = None
        self._forward_hook: Any = None
        self._total_flop: float = 0
        self._flops: float = 0
        self._start_time: float = 0
        self._profiled: bool = False

    def _on_trace_ready(self, prof: torch.profiler.profile) -> None:
        self._total_flop += sum(
            e.flops for e in prof.key_averages() if e.flops
        )

    def start(self) -> None:
        self._total_flop = 0
        self._flops = 0
        self._start_time = time.monotonic()
        self._prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=True,
            profile_memory=False,
            schedule=torch.profiler.schedule(
                wait=0, warmup=0, active=1, repeat=0
            ),
            on_trace_ready=self._on_trace_ready,
        )
        self._prof.start()
        self._forward_hook = self._model.register_forward_hook(self._on_forward)

    def step(self) -> None:
        if self._prof is not None:
            self._prof.step()

    def stop(self) -> None:
        elapsed = time.monotonic() - self._start_time
        if self._forward_hook is not None:
            self._forward_hook.remove()
            self._forward_hook = None
        if self._prof is not None:
            self._prof.stop()
            self._prof = None
            if self._total_flop == 0:
                log_warning(
                    "FLOP profiler recorded 0 FLOP. "
                    "Quantized kernels (e.g. bnb::matmul_4bit) may bypass aten::mm."
                )
            else:
                self._flops = self._total_flop / elapsed if elapsed > 0 else 0
                log_info(
                    "total_flop: %.4e, wall_time: %.2fs, flops: %.4e",
                    self._total_flop,
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
        entry = {"total_flop": self._total_flop, "flops": self._flops}
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

    def _on_forward(self, _module: torch.nn.Module, _args: Any, _output: Any) -> None:
        if self._forward_hook is not None:
            self._forward_hook.remove()
            self._forward_hook = None

    def _before_epoch(self, **kwargs: Any) -> None:
        if not self._profiled:
            self.start()

    def _after_batch(self, **kwargs: Any) -> None:
        self.step()

    def _after_epoch(self, epoch: int, **kwargs: Any) -> None:
        if not self._profiled and self._prof is not None:
            self.stop()
            self._set_epoch_metric(epoch, "total_flop", self._total_flop)
            self._set_epoch_metric(epoch, "flops", self._flops)
