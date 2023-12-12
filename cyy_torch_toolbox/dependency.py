import importlib.util

has_pynvml: bool = importlib.util.find_spec("pynvml") is not None
