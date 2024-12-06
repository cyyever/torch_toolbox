import os
import threading

from cyy_naive_lib.fs.ssd import is_ssd
from cyy_naive_lib.log import log_warning
from cyy_naive_lib.system_info import OSType, get_operating_system_type


class DatasetCache:
    __dataset_root_dir: str = os.path.join(os.path.expanduser("~"), "pytorch_dataset")
    lock = threading.RLock()

    @classmethod
    def __get_dataset_root_dir(cls) -> str:
        with cls.lock:
            return os.getenv("PYTORCH_DATASET_ROOT_DIR", cls.__dataset_root_dir)

    @classmethod
    def set_dataset_root_dir(cls, root_dir: str) -> None:
        with cls.lock:
            cls.__dataset_root_dir = root_dir

    @classmethod
    def get_dataset_dir(cls, name: str) -> str:
        dataset_dir = os.path.join(cls.__get_dataset_root_dir(), name)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        if get_operating_system_type() != OSType.Windows and not is_ssd(dataset_dir):
            log_warning("dataset %s is not on a SSD disk: %s", name, dataset_dir)
        return dataset_dir

    @classmethod
    def get_dataset_cache_dir(cls, name: str) -> str:
        cache_dir = os.path.join(cls.get_dataset_dir(name), ".cache")
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
