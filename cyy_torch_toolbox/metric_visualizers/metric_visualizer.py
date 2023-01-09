from cyy_torch_toolbox.hook import Hook


class MetricVisualizer(Hook):
    def __init__(self, **kwargs):
        super().__init__(stripable=True, **kwargs)
        self._prefix = None
        self._data_dir = None

    def set_data_dir(self, data_dir: str) -> None:
        self._data_dir = data_dir

    def set_prefix(self, prefix: str) -> None:
        self._prefix = prefix

    @property
    def prefix(self):
        return self._prefix
