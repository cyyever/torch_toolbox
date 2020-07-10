import copy
import tempfile

from .util import split_list_to_chunks
from .dataset import sub_dataset
from .device import get_device
from .hyper_gradient_trainer import HyperGradientTrainer


class HyperGradientAnalyzer:
    def __init__(self, validator, hyper_gradient_dir):
        assert validator.loss_fun.reduction in ("mean", "elementwise_mean")
        self.validator = validator

        self.cache_size = 1024
        self.hyper_gradient_matrix = HyperGradientTrainer.create_gradient_matrix(
            self.cache_size, self.validator.model, hyper_gradient_dir)
        self.contributions = None

    def get_contributions(self, training_set_size=None):
        if training_set_size is None:
            training_set_size = len(self.hyper_gradient_matrix)
        contribution_dict = dict()
        validation_gradient = self.validator.get_gradient()
        for chunk in split_list_to_chunks(
            self.hyper_gradient_matrix.keys(), self.cache_size
        ):
            self.hyper_gradient_matrix.prefetch(chunk)
            for instance_index in chunk:
                hyper_gradient = self.hyper_gradient_matrix[instance_index].to(
                    get_device()
                )
                contribution_dict[int(instance_index)] = (
                    -(validation_gradient @ hyper_gradient).data.item()
                    / training_set_size
                )
        assert len(contribution_dict) == training_set_size
        return contribution_dict

    def get_subset_contributions(
            self,
            training_subset_dict,
            validation_subset_dict,
            training_set_size=None):
        if training_set_size is None:
            training_set_size = len(self.hyper_gradient_matrix)
        hyper_gradient_sum_dict = HyperGradientTrainer.create_gradient_matrix(
            self.cache_size, self.validator.model
        )
        hyper_gradient_sum_dict.set_storage_dir(tempfile.gettempdir())

        for k, indices in training_subset_dict.items():
            chunk = [str(index) for index in indices]
            self.hyper_gradient_matrix.prefetch(chunk)
            hyper_gradient_sum = None
            for instance_index in chunk:
                hyper_gradient = self.hyper_gradient_matrix[instance_index].to(
                    get_device()
                )
                if hyper_gradient_sum is None:
                    hyper_gradient_sum = hyper_gradient
                else:
                    hyper_gradient_sum += hyper_gradient
                hyper_gradient_sum_dict[str(k)] = hyper_gradient_sum
        tmp_validator = copy.deepcopy(self.validator)
        contribution_dict = dict()
        for k, indices in validation_subset_dict.items():
            subset = sub_dataset(self.validator.dataset, indices)
            assert len(subset) == len(indices)
            tmp_validator.set_dataset(subset)
            sub_validator_gradient = tmp_validator.get_gradient() * len(indices)
            for k2 in hyper_gradient_sum_dict.keys():
                gradient_sum = hyper_gradient_sum_dict[k2]
                k2 = int(k2)
                if k2 not in contribution_dict:
                    contribution_dict[k2] = dict()
                contribution_dict[k2][k] = (
                    -(sub_validator_gradient @ gradient_sum) / training_set_size
                ).data.item()
        return contribution_dict
