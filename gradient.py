import copy
from synced_tensor_dict_util import create_tensor_dict


def get_dataset_gradients(dataset_dict: dict, validator, cache_size=512):
    tensor_dict = create_tensor_dict(cache_size)
    for k, dataset in dataset_dict.items():
        tmp_validator = copy.deepcopy(validator)
        tmp_validator.set_dataset(dataset)
        tensor_dict[k] = tmp_validator.get_gradient()
    return tensor_dict
