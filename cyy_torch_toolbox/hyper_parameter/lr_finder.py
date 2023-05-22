import torch

from ..hook import Hook
from ..ml_type import StopExecutingException


class LRFinder(Hook):
    "Training with exponentially growing learning rate, coped from fastai"

    def __init__(self, start_lr=1e-7, end_lr=10, epoch=2, stop_div=True):
        super().__init__()
        self.lr_getter = lambda idx: start_lr * (end_lr / start_lr) ** idx
        self.epoch = epoch
        self.stop_div = stop_div
        self.best_loss = float("inf")
        self.losses = []
        self.learning_rates = []
        self.batch_index = 0
        self.total_batch_num = None
        self.suggested_learning_rate = None

    def _before_execute(self, **kwargs):
        trainer = kwargs["executor"]
        trainer.remove_optimizer()
        trainer.remove_lr_scheduler()
        trainer.hyper_parameter.epoch = self.epoch
        trainer.hyper_parameter.learning_rate = 1
        self.total_batch_num = self.epoch * (
            (trainer.dataset_size + trainer.hyper_parameter.batch_size - 1)
            // trainer.hyper_parameter.batch_size
        )

    def _before_batch(self, **kwargs):
        trainer = kwargs["executor"]
        learning_rate = self.lr_getter(self.batch_index / (self.total_batch_num - 1))
        self.learning_rates.append(learning_rate)
        optimizer = trainer.get_optimizer()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    def _after_batch(self, **kwargs):
        batch_loss = kwargs["result"]["loss"]
        if self.losses:
            batch_loss = batch_loss + 0.98 * (self.losses[-1] - batch_loss)
        self.losses.append(batch_loss)

        if batch_loss < self.best_loss:
            self.best_loss = batch_loss

        stop_training = False
        if batch_loss > 10 * self.best_loss and kwargs["epoch"] > 1 and self.stop_div:
            stop_training = True
        if self.batch_index + 1 == self.total_batch_num:
            stop_training = True

        self.batch_index += 1
        if stop_training:
            self.learning_rates = self.learning_rates[self.total_batch_num // 10:]
            self.losses = self.losses[self.total_batch_num // 10:]
            self.suggested_learning_rate = (
                self.learning_rates[torch.tensor(self.losses).argmin()] / 10.0
            )
            raise StopExecutingException()