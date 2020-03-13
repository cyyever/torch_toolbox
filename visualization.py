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

    def __init__(self, title, env="main"):
        self.vis = visdom.Visdom(env=env)
        self.win = None
        self.title = title

    def plot_learning_rate(self, epoch, learning_rate):
        self.__plot_line(
            torch.LongTensor([epoch]),
            "Epoch",
            torch.Tensor([learning_rate]),
            "Learning Rate",
            None,
        )

    def plot_loss(self, epoch, loss, name):
        self.__plot_line(
            torch.LongTensor(
                [epoch]), "Epoch", torch.Tensor(
                [loss]), "Loss", name)

    def plot_accuracy(self, epoch, accuracy, name):
        self.__plot_line(
            torch.LongTensor([epoch]),
            "Epoch",
            torch.Tensor([accuracy]),
            "Accuracy",
            name,
        )

    def __plot_line(self, x, x_label, y, y_label, name):
        update = None

        if self.win is not None and not self.vis.win_exists(self.win):
            self.win = None

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
