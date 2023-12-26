from typing import TypeAlias

import torch

TensorDict: TypeAlias = dict[str, torch.Tensor]
