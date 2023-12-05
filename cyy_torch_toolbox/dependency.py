import importlib.util

has_torchvision: bool = importlib.util.find_spec("torchvision") is not None
has_torch_geometric: bool = (
    importlib.util.find_spec("torch_geometric") is not None
    and importlib.util.find_spec("pyg_lib") is not None
)
has_pynvml: bool = importlib.util.find_spec("pynvml") is not None
