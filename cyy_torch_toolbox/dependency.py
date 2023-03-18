try:
    import torchvision

    has_torchvision = True
except BaseException:
    has_torchvision = False

try:
    import torchtext

    has_torchtext = True
except BaseException:
    has_torchtext = False

try:
    import torch_geometric

    has_torch_geometric = True
except BaseException:
    has_torch_geometric = False
try:
    import torchaudio

    has_torchaudio = True
except BaseException:
    has_torchaudio = False


try:
    import medmnist

    has_medmnist = True
except BaseException:
    has_medmnist = False

try:
    import datasets
    import transformers

    has_hugging_face = True
except BaseException:
    has_hugging_face = False
