import shutil

import torch

from data_structure.synced_tensor_dict import SyncedTensorDict

tensor_dict = SyncedTensorDict.create(int, 10, storage_dir="tensor_dict_dir")

for i in range(100):
    tensor_dict[i] = torch.Tensor([i])

for (key, tensor) in tensor_dict.iterate():
    assert tensor == torch.Tensor([key])

for (key, tensor) in tensor_dict.iterate({"1", "2"}):
    assert 1 <= key <= 2
    assert tensor == torch.Tensor([key])

tensor_dict.release()
shutil.rmtree("tensor_dict_dir")
