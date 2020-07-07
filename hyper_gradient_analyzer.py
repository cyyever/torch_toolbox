import copy
import torch
import torch.nn.utils.prune as prune
from cyy_naive_lib.log import get_logger

from .util import (
    model_parameters_to_vector,
    get_pruning_mask,
    split_chunks,
)
from .device import get_device
from .hyper_gradient_trainer import HyperGradientTrainer


class HyperGradientAnalyzer:
    def __init__(self, validator, hyper_gradient_dir):
        self.validator = validator
        self.cache_size = 1024
        self.hyper_gradient_matrix = self.__load_hyper_gradients(
            hyper_gradient_dir, self.cache_size
        )
        self.contributions = None

    def get_contributions(self, training_set_size=None):
        if training_set_size is None:
            training_set_size = len(self.hyper_gradient_matrix)
        contribution_dict = dict()
        validation_gradient = self.validator.get_gradient()
        for chunk in split_chunks(
                self.hyper_gradient_matrix.keys(),
                self.cache_size):
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
        hyper_gradient_sum_dict = dict()
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
                hyper_gradient_sum_dict[k] = hyper_gradient_sum
        tmp_validator = copy.deepcopy(self.validator)
        contribution_dict = dict()
        for k, indices in validation_subset_dict.items():
            tmp_validator.set_dataset(
                torch.utils.data.Subset(self.validator.dataset, indices)
            )
            sub_validator_gradient = tmp_validator.get_gradient()
            for k2, gradient_sum in hyper_gradient_sum_dict.items():
                contribution_dict[(k2, k)] = (
                    -(sub_validator_gradient @ gradient_sum) / training_set_size
                ).data.item()
        return contribution_dict

    def __load_hyper_gradients(self, hyper_gradient_dir, cache_size):
        model = self.validator.model
        mask = None
        gradient_shape = None
        if prune.is_pruned(model):
            get_logger().info("use pruned model")
            parameters = model_parameters_to_vector(model)
            gradient_shape = parameters.shape
            mask = get_pruning_mask(model)
            assert len(mask) == len(parameters)
        return HyperGradientTrainer.create_gradient_matrix(
            cache_size, mask, gradient_shape, hyper_gradient_dir
        )
