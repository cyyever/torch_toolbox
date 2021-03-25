import copy
import tempfile

from data_structure.synced_tensor_dict import SyncedTensorDict
from inference import Inferencer


def get_dataset_gradients(dataset_dict: dict, inferencer: Inferencer, cache_size=512):
    tensor_dict = SyncedTensorDict.create(cache_size=cache_size)
    tensor_dict.set_storage_dir(tempfile.gettempdir())
    for k, dataset in dataset_dict.items():
        tmp_inferencer = copy.deepcopy(inferencer)
        tmp_inferencer.set_dataset(dataset)
        tensor_dict[str(k)] = tmp_inferencer.get_gradient()
    return tensor_dict
