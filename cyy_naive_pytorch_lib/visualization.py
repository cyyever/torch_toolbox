import torch
import visdom


class Window:
    __envs: dict = dict()
    __sessions: dict = {}

    @staticmethod
    def save_envs():
        visdom.Visdom().save(list(Window.__envs.keys()))

    def __init__(self, title, env=None):
        if env is None:
            env = "main"
        if env not in Window.__sessions:
            Window.__sessions[env] = visdom.Visdom(env=env)

        self.vis = Window.__sessions[env]
        self.title = title
        self.win = None
        if env in Window.__envs:
            self.win = Window.__envs[env].get(title, None)
        self.x_label = None
        self.y_label = None
        self.extra_opts = dict(ytick=True)
        self.showlegend = True

    def set_opt(self, k: str, v):
        self.extra_opts[k] = v

    def get_opts(self):
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
        opts.update(self.get_opts())
        self.win = self.vis.line(
            Y=y,
            X=x,
            win=self.win,
            name=name,
            update=update,
            opts=opts,
        )
        self._add_window()

    def plot_histogram(self, tensor):
        opts = dict(numbins=1024)
        opts.update(self.get_opts())
        self.win = self.vis.histogram(tensor.view(-1), win=self.win, opts=opts)

    def plot_scatter(self, x, y=None, name=None):
        update = None
        if self.win is not None:
            update = "replace"
        self.win = self.vis.scatter(
            X=x, Y=y, win=self.win, name=name, update=update, opts=self.get_opts()
        )
        self._add_window()

    def save(self):
        self.vis.save([self.vis.env])

    def _add_window(self):
        if self.vis.env not in Window.__envs:
            Window.__envs[self.vis.env] = dict()
        Window.__envs[self.vis.env][self.title] = self.win


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
