import torch

from ml_types import ModelExecutorCallbackPoint, StopExecutingException
from visualization import BatchWindow


class LRFinder:
    "Training with exponentially growing learning rate, coped from fastai"

    def __init__(self, start_lr=1e-7, end_lr=10, epoch=2, stop_div=True):
        self.lr_getter = lambda idx: start_lr * (end_lr / start_lr) ** idx
        self.epoch = epoch
        self.stop_div = stop_div
        self.best_loss = float("inf")
        self.losses = []
        self.learning_rates = []
        self.total_batch_num = None
        self.suggested_learning_rate = None

    def add_callback(self, trainer):
        trainer.add_callback(
            ModelExecutorCallbackPoint.BEFORE_TRAINING,
            self.before_training,
        )
        trainer.add_callback(
            ModelExecutorCallbackPoint.BEFORE_BATCH,
            self.before_batch,
        )
        trainer.add_callback(
            ModelExecutorCallbackPoint.AFTER_BATCH,
            self.after_batch,
        )

    def before_training(self, trainer):
        trainer.remove_optimizer()
        trainer.remove_lr_scheduler()
        trainer.hyper_parameter.set_learning_rate(1)
        print("prepare finding")
        self.total_batch_num = (
            len(trainer.training_dataset) + trainer.hyper_parameter.batch_size - 1
        ) // trainer.hyper_parameter.batch_size

    def before_batch(self, trainer, batch_index, **kwargs):
        optimizer = trainer.get_optimizer()
        for group in optimizer.param_groups:
            group["lr"] = self.lr_getter(batch_index / self.total_batch_num - 1)

    def after_batch(self, trainer, batch_index, **kwargs):
        batch_loss = kwargs["batch_loss"]
        learning_rate = trainer.get_data("cur_learning_rates")[0]
        BatchWindow(
            "LRFinder learning rate", env=trainer.visdom_env
        ).plot_learning_rate(batch_index, learning_rate)
        BatchWindow("LRFinder batch loss", env=trainer.visdom_env).plot_learning_rate(
            batch_index, batch_loss
        )
        self.learning_rates.append(learning_rate)
        self.losses.append(batch_loss)

        if batch_loss < self.best_loss:
            self.best_loss = batch_loss

        stop_training = False
        if batch_loss > 4 * self.best_loss and self.stop_div:
            stop_training = True
        if batch_index + 1 == self.total_batch_num:
            stop_training = True

        if stop_training:
            self.learning_rates = torch.Tensor(
                self.learning_rates[self.total_batch_num // 10:]
            )
            self.losses = torch.Tensor(self.losses[self.total_batch_num // 10:])
            self.suggested_learning_rate = (
                self.learning_rates[self.losses.argmin()].item() / 10.0
            )
            raise StopExecutingException()
