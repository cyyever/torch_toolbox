import os
import shutil
import torch
import cyy_pytorch_cpp
from synced_tensor_dict_util import iterate_over_synced_tensor_dict

tensor_dict = cyy_pytorch_cpp.data_structure.SyncedTensorDict("")
tensor_dict.set_in_memory_number(10)
tensor_dict.set_storage_dir("tensor_dict_dir")

for i in range(100):
    tensor_dict[str(i)] = torch.Tensor([i])

for (key, tensor) in iterate_over_synced_tensor_dict(tensor_dict):
    assert tensor == torch.Tensor([int(key)])

for (key, tensor) in iterate_over_synced_tensor_dict(tensor_dict, {"1", "2"}):
    assert 1 <= int(key) <= 2
    assert tensor == torch.Tensor([int(key)])

del tensor_dict
shutil.rmtree("tensor_dict_dir")
