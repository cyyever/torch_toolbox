from pathlib import Path
from typing import Any

from ..hook import Hook


class MetricVisualizer(Hook):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(stripable=True, **kwargs)
        self._prefix: str = ""
        self._data_dir: Path | None = None

    def set_data_dir(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)

    def set_prefix(self, prefix: str) -> None:
        self._prefix = prefix

    @property
    def data_dir(self) -> Path:
        assert self._data_dir is not None
        return self._data_dir

    @property
    def prefix(self) -> str:
        return self._prefix
