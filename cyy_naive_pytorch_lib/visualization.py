import torch
import visdom


class Window:
    __envs: dict = dict()
    __windows: dict = dict()

    @staticmethod
    def save_envs():
        for env, vis in Window.__envs.items():
            vis.save([env])

    def __init__(self, title, env=None):
        if env is None:
            env = "main"
        self.env = env
        self.title = title
        self.x_label = None
        self.y_label = None
        self.extra_opts = dict(ytick=True)
        self.showlegend = True

    @property
    def vis(self):
        if self.env not in Window.__envs:
            Window.__envs[self.env] = visdom.Visdom(env=self.env)
        return Window.__envs[self.env]

    @property
    def win(self):
        if self.env not in Window.__windows:
            Window.__windows[self.env] = dict()
        return Window.__windows[self.env].get(self.title, None)

    def set_opt(self, k: str, v):
        self.extra_opts[k] = v

    def __get_opts(self):
        opts = dict(title=self.title, showlegend=self.showlegend)
        opts.update(self.extra_opts)
        return opts

    def plot_line(self, x, y, name=None):
        assert self.x_label
        assert self.y_label

        update = None
        if self.win is not None:
            if x.shape[0] == 1:
                update = "append"
            else:
                update = "replace"
        if name is None:
            name = self.y_label

        opts = dict(
            xlabel=self.x_label,
            ylabel=self.y_label,
        )
        opts.update(self.__get_opts())
        win = self.vis.line(
            Y=y,
            X=x,
            win=self.win,
            name=name,
            update=update,
            opts=opts,
        )
        self.__add_window(win)

    def plot_histogram(self, tensor):
        opts = dict(numbins=1024)
        opts.update(self.__get_opts())
        win = self.vis.histogram(tensor.view(-1), win=self.win, opts=opts)
        self.__add_window(win)

    def plot_scatter(self, x, y=None, name=None):
        update = None
        if self.win is not None:
            update = "replace"
        win = self.vis.scatter(
            X=x, Y=y, win=self.win, name=name, update=update, opts=self.__get_opts()
        )
        self.__add_window(win)

    def save(self):
        self.vis.save([self.env])

    def __add_window(self, win):
        if self.win is None:
            Window.__windows[self.env][self.title] = win


class IterationWindow(Window):
    def plot_learning_rate(self, epoch, learning_rate):
        self.y_label = "Learning Rate"
        self.plot_scalar(epoch, learning_rate)

    def plot_loss(self, epoch, loss, name=None):
        self.y_label = "Loss"
        return self.plot_scalar(epoch, loss, name=name)

    def plot_accuracy(self, epoch, accuracy, name=None):
        self.y_label = "Accuracy"
        return self.plot_scalar(epoch, accuracy, name=name)

    def plot_scalar(self, epoch, scalar, name=None):
        super().plot_line(
            torch.LongTensor([epoch]),
            torch.Tensor([scalar]),
            name=name,
        )


class EpochWindow(IterationWindow):
    def __init__(self, title, env=None):
        super().__init__(title, env=env)
        self.x_label = "Epoch"


class BatchWindow(IterationWindow):
    def __init__(self, title, env=None):
        super().__init__(title, env=env)
        self.x_label = "Batch"
