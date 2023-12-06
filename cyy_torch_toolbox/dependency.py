import importlib.util

has_torchvision: bool = importlib.util.find_spec("torchvision") is not None
has_pynvml: bool = importlib.util.find_spec("pynvml") is not None
