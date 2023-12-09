from cyy_torch_toolbox.hook import Hook


class MetricVisualizer(Hook):
    def __init__(self, **kwargs) -> None:
        super().__init__(stripable=True, **kwargs)
        self._prefix: str = ""
        self._data_dir: None | str = None

    def set_data_dir(self, data_dir: str) -> None:
        self._data_dir = data_dir

    def set_prefix(self, prefix: str) -> None:
        self._prefix = prefix

    @property
    def data_dir(self) -> None | str:
        return self._data_dir

    @property
    def prefix(self) -> str:
        return self._prefix
