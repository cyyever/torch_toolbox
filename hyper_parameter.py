import torch


class HyperParameter:
    def __init__(self):
        self.learning_rate = None
        self.weight_decay = None
        self.lr_scheduler_factory = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=(lambda _: 1))
        self.optimizer_factory = None

    def set_lr_scheduler_factory(self, lr_scheduler_factory):
        self.lr_scheduler_factory = lr_scheduler_factory

    def get_lr_scheduler(self, optimizer):
        return self.lr_scheduler_factory(optimizer)

    def set_optimizer_factory(self, optimizer_factory):
        self.optimizer_factory = optimizer_factory

    def get_optimizer(self, params):
        return self.optimizer_factory(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
