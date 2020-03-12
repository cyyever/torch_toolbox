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

    def plot_loss(self, loss, name):
        if isinstance(loss, list):
            loss = torch.Tensor(loss)
            epoches = torch.arange(len(loss), dtype=torch.int64)
        elif isinstance(loss, dict):
            keys = sorted(loss.keys())
            epoches = torch.LongTensor(keys)
            loss = torch.Tensor([loss[k] for k in keys])
        update = None

        if self.win is not None and not self.vis.win_exists(self.win):
            self.win = None

        if self.win is not None:
            update = "replace"

        self.win = self.vis.line(
            Y=loss,
            X=epoches,
            win=self.win,
            name=name,
            update=update,
            opts=dict(xlabel="Epoch", ylabel="Loss", title=self.title),
        )
