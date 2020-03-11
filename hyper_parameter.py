import torch


class HyperParameter:
    def __init__(self, epoches, batch_size, learning_rate, weight_decay=0):
        self.epoches = epoches
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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
            params, self.learning_rate, self.weight_decay)
