import os
import threading
from pathlib import Path

from cyy_naive_lib.fs.ssd import is_ssd
from cyy_naive_lib.log import log_warning
from cyy_naive_lib.system_info import OSType, get_operating_system_type


class DatasetCache:
    __dataset_root_dir: Path = Path.home() / "pytorch_dataset"
    lock = threading.RLock()

    @classmethod
    def __get_dataset_root_dir(cls) -> Path:
        with cls.lock:
            env_dir = os.getenv("PYTORCH_DATASET_ROOT_DIR")
            if env_dir is not None:
                return Path(env_dir)
            return cls.__dataset_root_dir

    @classmethod
    def set_dataset_root_dir(cls, root_dir: str | Path) -> None:
        with cls.lock:
            cls.__dataset_root_dir = Path(root_dir)

    @classmethod
    def get_dataset_dir(cls, name: str) -> Path:
        dataset_dir = cls.__get_dataset_root_dir() / name
        if not dataset_dir.is_dir():
            dataset_dir.mkdir(parents=True, exist_ok=True)
        if get_operating_system_type() != OSType.Windows and not is_ssd(dataset_dir):
            log_warning("dataset %s is not on a SSD disk: %s", name, dataset_dir)
        return dataset_dir

    @classmethod
    def get_dataset_cache_dir(cls, name: str) -> Path:
        with cls.lock:
            cache_dir = cls.get_dataset_dir(name) / ".cache"
            if not cache_dir.is_dir():
                cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir
