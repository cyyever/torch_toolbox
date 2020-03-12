import visdom
import torch


class Visualization:
    def __init__(self, env="main"):
        self.vis = visdom.Visdom(env=env)
        self.win = None

    def set_title(self, title):
        layout = dict(title=title)
        self.win = self.vis._send({"layout": layout})

    def plot_loss(self, loss, name):
        if isinstance(loss, list):
            loss = torch.Tensor(loss)
            epoches = torch.arange(1, len(loss) + 1)
        elif isinstance(loss, dict):
            keys = sorted(loss.keys)
            print(keys, "aaaaaaaaaaa")
            epoches = torch.Tensor(keys)
            loss = torch.Tensor([loss[k] for k in keys])

        self.win = self.vis.line(
            Y=loss,
            X=epoches,
            win=self.win,
            name=name,
            update="append",
            opts=dict(xlabel="Epoch", ylabel="Loss"),
        )
