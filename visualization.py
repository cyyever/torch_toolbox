import visdom
import torch


class Window:
    cur_env = "main"
    envs: dict = {cur_env: dict()}

    @staticmethod
    def set_env(env):
        Window.cur_env = env

    @staticmethod
    def add_window(win):
        if win.env not in Window.envs:
            Window.envs[win.env] = dict()
        Window.envs[win.env][win.title] = win

    def __init__(self, title, env=None, x_label="", y_label=""):
        if env is None:
            env = Window.cur_env
        self.vis = visdom.Visdom(env=env)
        self.env = env
        self.title = title
        self.win = None
        if env in Window.envs:
            self.win = Window.envs[env].get(title, None)
        self.x_label = x_label
        self.y_label = y_label

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
            opts=dict(
                xlabel=x_label,
                ylabel=y_label,
                title=self.title,
                showlegend=True),
        )
        Window.add_window(self)

    def plot_histogram(self, tensor):
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
        Window.add_window(self)

    def save(self):
        self.vis.save([self.env])


class EpochWindow(Window):
    def __init__(self, title, env=None, y_label=""):
        super().__init__(title, env=env, x_label="Epoch", y_label=y_label)

    def plot_learning_rate(self, epoch, learning_rate):
        self.plot_scalar(epoch, learning_rate, y_label="Learning Rate")

    def plot_loss(self, epoch, loss, name=None):
        return self.plot_scalar(epoch, loss, y_label="Loss", name=name)

    def plot_accuracy(self, epoch, accuracy, name=None):
        return self.plot_scalar(epoch, accuracy, y_label="Accuracy", name=name)

    def plot_scalar(self, epoch, scalar, y_label=None, name=None):
        if y_label is None:
            y_label = self.y_label

        self.__plot_line(
            torch.LongTensor([epoch]),
            torch.Tensor([scalar]),
            y_label=y_label,
            name=name,
        )
