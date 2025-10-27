import copy
from typing import Any

import torch
from cyy_naive_lib.log import log_warning

from ..concurrency import TorchThreadTaskQueue
from ..hook import Hook
from ..ml_type import StopExecutingException


class LRFinder(Hook):
    "Training with exponentially growing learning rate, coped from fastai"

    def __init__(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        epoch: int = 2,
        stop_div: bool = True,
    ) -> None:
        super().__init__()
        self.lr_getter = lambda idx: start_lr * (end_lr / start_lr) ** idx
        self.epoch: int = epoch
        self.stop_div: bool = stop_div
        self.best_loss: float = float("inf")
        self.losses: list[float] = []
        self.learning_rates: list[float] = []
        self.batch_index: int = 0
        self.total_batch_num: int = 0
        self.suggested_learning_rate: float = 0

    def _before_execute(self, **kwargs: Any) -> None:
        trainer = kwargs["executor"]
        trainer.remove_optimizer()
        trainer.remove_lr_scheduler()
        trainer.hyper_parameter.epoch = self.epoch
        trainer.hyper_parameter.learning_rate = 1
        self.total_batch_num = self.epoch * (
            (trainer.dataset_size + trainer.hyper_parameter.batch_size - 1)
            // trainer.hyper_parameter.batch_size
        )

    def _before_batch(self, **kwargs: Any) -> None:
        trainer = kwargs["executor"]
        learning_rate = self.lr_getter(self.batch_index / (self.total_batch_num - 1))
        self.learning_rates.append(learning_rate)
        optimizer = trainer.get_optimizer()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    def _after_batch(self, **kwargs: Any) -> None:
        batch_loss = kwargs["result"]["loss"].clone()
        if self.losses:
            batch_loss = batch_loss + 0.98 * (self.losses[-1] - batch_loss)
        self.losses.append(batch_loss)

        self.best_loss = min(batch_loss, self.best_loss)

        stop_training = False
        if batch_loss > 10 * self.best_loss and kwargs["epoch"] > 1 and self.stop_div:
            stop_training = True
        self.batch_index += 1
        if self.batch_index == self.total_batch_num:
            stop_training = True

        if stop_training:
            self.learning_rates = self.learning_rates[self.total_batch_num // 10 :]
            self.losses = self.losses[self.total_batch_num // 10 :]
            self.suggested_learning_rate = (
                self.learning_rates[torch.tensor(self.losses).argmin()] / 10.0
            )
            raise StopExecutingException()


def __determine_learning_rate(task: Any, **kwargs: Any) -> float:
    tmp_trainer = task
    with tmp_trainer.hook_config:
        tmp_trainer.hook_config.use_amp = False
        tmp_trainer.hook_config.disable_log()
        tmp_trainer.disable_stripable_hooks()
        lr_finder = LRFinder()
        log_warning("register lr_finder %s", id(tmp_trainer))
        tmp_trainer.prepend_hook(lr_finder)
        tmp_trainer.train()
        log_warning("suggested_learning_rate is %s", lr_finder.suggested_learning_rate)
        assert lr_finder.suggested_learning_rate is not None
        return lr_finder.suggested_learning_rate


def get_learning_rate(trainer: Any) -> float:
    task_queue = TorchThreadTaskQueue()
    task_queue.start(worker_fun=__determine_learning_rate)
    trainer.offload_from_device()
    task_queue.add_task(copy.deepcopy(trainer))
    data = task_queue.get_data()
    assert data.is_ok()
    learning_rate = data.value()
    assert isinstance(learning_rate, float)
    task_queue.stop()
    return learning_rate
