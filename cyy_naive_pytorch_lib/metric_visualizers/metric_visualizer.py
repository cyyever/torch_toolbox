from callback import Callback
from metrics.metric import Metric


class MetricVisualizer(Callback):
    def __init__(self, metric: Metric = None):
        super().__init__()
        self.__metric = metric

    @property
    def metric(self):
        return self.__metric
