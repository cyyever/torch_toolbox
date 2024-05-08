from typing import Iterable, TypeAlias

import torch

OptionalTensor: TypeAlias = torch.Tensor | None
TensorDict: TypeAlias = dict[str, torch.Tensor]
OptionalTensorDict: TypeAlias = TensorDict | None
BlockType: TypeAlias = list[tuple[str, torch.nn.Module]]
IndicesType: TypeAlias = Iterable[int]
OptionalIndicesType: TypeAlias = IndicesType | None
