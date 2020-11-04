class HyperParameter:
    def __init__(self, epochs, batch_size, learning_rate, weight_decay=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_factory = None
        self.optimizer_factory = None

    def set_lr_scheduler_factory(self, lr_scheduler_factory):
        self.lr_scheduler_factory = lr_scheduler_factory

    def get_lr_scheduler(self, optimizer, training_dataset_size):
        return self.lr_scheduler_factory(
            optimizer, self, training_dataset_size)

    def set_optimizer_factory(self, optimizer_factory):
        self.optimizer_factory = optimizer_factory

    def get_optimizer(self, params, training_dataset):
        return self.optimizer_factory(
            params, self.learning_rate, self.weight_decay, training_dataset
        )

    def __str__(self):
        s = (
            "epochs:"
            + str(self.epochs)
            + " batch_size:"
            + str(self.batch_size)
            + " learning_rate:"
            + str(self.learning_rate)
            + " weight_decay:"
            + str(self.weight_decay)
        )
        # if self.lr_scheduler_factory is not None:
        #     s += str(self.lr_scheduler_factory)

        # if self.optimizer_factory is not None:
        #     s += str(self.optimizer_factory)
        return s
