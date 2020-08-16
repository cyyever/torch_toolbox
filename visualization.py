import torch
import visdom


class Window:
    __envs: dict = dict()
    __sessions: dict = {}

    @staticmethod
    def save_envs():
        visdom.Visdom().save(list(Window.__envs.keys()))

    def __init__(
            self,
            title,
            env=None,
            x_label="",
            y_label="",
            showlegend=True):
        if env is None:
            env = "main"
        if env not in Window.__sessions:
            Window.__sessions[env] = visdom.Visdom(env=env)

        self.vis = Window.__sessions[env]
        self.title = title
        self.win = None
        if env in Window.__envs:
            self.win = Window.__envs[env].get(title, None)
        self.x_label = x_label
        self.y_label = y_label
        self.opts = dict(ytick=True)
        self.showlegend = showlegend

    def set_opt(self, k: str, v):
        self.opts[k] = v

    def plot_line(self, x, y, x_label=None, y_label=None, name=None):

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
            opts=(
                dict(
                    xlabel=x_label,
                    ylabel=y_label,
                    title=self.title,
                    showlegend=self.showlegend,
                )
                | self.opts
            ),
        )
        self._add_window()

    def plot_histogram(self, tensor):
        self.win = self.vis.histogram(
            tensor.view(-1),
            win=self.win,
            opts=(dict(numbins=1024, title=self.title) | self.opts),
        )

    def plot_scatter(self, x, y=None, name=None):
        update = None
        if self.win is not None:
            update = "replace"
        self.win = self.vis.scatter(
            X=x,
            Y=y,
            win=self.win,
            name=name,
            update=update,
            opts=(
                dict(
                    title=self.title,
                    showlegend=self.showlegend) | self.opts),
        )
        self._add_window()

    def save(self):
        self.vis.save([self.vis.env])

    def _add_window(self):
        if self.vis.env not in Window.__envs:
            Window.__envs[self.vis.env] = dict()
        Window.__envs[self.vis.env][self.title] = self.win


class EpochWindow(Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_learning_rate(self, epoch, learning_rate):
        self.plot_scalar(epoch, learning_rate, y_label="Learning Rate")

    def plot_loss(self, epoch, loss, name=None):
        return self.plot_scalar(epoch, loss, y_label="Loss", name=name)

    def plot_accuracy(self, epoch, accuracy, name=None):
        return self.plot_scalar(epoch, accuracy, y_label="Accuracy", name=name)

    def plot_scalar(self, epoch, scalar, y_label=None, name=None):
        if y_label is None:
            y_label = self.y_label

        super().plot_line(
            torch.LongTensor([epoch]),
            torch.Tensor([scalar]),
            y_label=y_label,
            name=name,
        )
