import visdom
import torch


class Window:
    instances = dict()

    @staticmethod
    def get(title):
        if title not in Window.instances:
            instance = Window(title)
            Window.instances[title] = instance
        return Window.instances[title]

    def __init__(self, title, x_label="", y_label="", env="main"):
        self.vis = visdom.Visdom(env=env)
        self.win = None
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    def plot_learning_rate(self, epoch, learning_rate):
        self.__plot_line(
            torch.LongTensor([epoch]),
            torch.Tensor([learning_rate]),
            "Epoch",
            "Learning Rate",
            None,
        )

    def plot_scalar_by_epoch(self, epoch, scalar, name=None):
        self.__plot_line(
            torch.LongTensor([epoch]), torch.Tensor([scalar]), "Epoch", name,
        )

    def plot_loss(self, epoch, loss, name=None):
        self.y_label = "Loss"
        return self.plot_scalar_by_epoch(epoch, loss, name)

    def plot_accuracy(self, epoch, accuracy, name=None):
        self.y_label = "Accuracy"
        return self.plot_scalar_by_epoch(epoch, accuracy, name)

    def __plot_line(self, x, y, x_label=None, y_label=None, name=None):
        if self.win is not None and not self.vis.win_exists(self.win):
            self.win = None

        if x_label is None:
            x_label = self.x_label

        if y_label is None:
            y_label = self.y_label

        update = None
        if self.win is not None:
            if x.shape[0] == 1:
                update = "append"
            else:
                update = "replace"
        if name is None:
            name = y_label

        self.win = self.vis.line(
            Y=y,
            X=x,
            win=self.win,
            name=name,
            update=update,
            opts=dict(xlabel=x_label, ylabel=y_label, title=self.title),
        )

    def plot_histogram(self, tensor, name=None):
        if self.win is not None and not self.vis.win_exists(self.win):
            self.win = None

        self.win = self.vis.histogram(
            tensor.view(-1), win=self.win, opts=dict(numbins=1024, title=self.title)
        )

    def plot_scatter(self, tensor, name=None):
        if self.win is not None and not self.vis.win_exists(self.win):
            self.win = None
        update = None
        if self.win is not None:
            update = "replace"
        self.win = self.vis.scatter(
            tensor,
            win=self.win,
            name=name,
            update=update,
            opts=dict(
                title=self.title))
