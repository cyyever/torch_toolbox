import importlib.util

has_torchvision: bool = importlib.util.find_spec("torchvision") is not None
has_torchtext: bool = importlib.util.find_spec("torchtext") is not None
has_torch_geometric: bool = (
    importlib.util.find_spec("torch_geometric") is not None
    and importlib.util.find_spec("pyg_lib") is not None
)
has_torchaudio: bool = importlib.util.find_spec("torchaudio") is not None
has_hugging_face: bool = (
    importlib.util.find_spec("datasets") is not None
    and importlib.util.find_spec("transformers") is not None
)
has_dali: bool = (
    importlib.util.find_spec("nvidia") is not None
    and importlib.util.find_spec("nvidia.dali") is not None
)
has_spacy: bool = importlib.util.find_spec("spacy") is not None
has_pynvml: bool = importlib.util.find_spec("pynvml") is not None
