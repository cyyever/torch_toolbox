from .metric_visualizer import MetricVisualizer


class MetricLogger(MetricVisualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefix = ""

    def set_prefix(self, prefix: str):
        self.prefix = prefix
