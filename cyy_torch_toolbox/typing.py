from typing import TypeAlias

import torch

TensorDict: TypeAlias = dict[str, torch.Tensor]
BlockType: TypeAlias = list[tuple[str, torch.nn.Module]]
