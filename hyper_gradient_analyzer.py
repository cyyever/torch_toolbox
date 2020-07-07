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
    def __init__(self, training_dataset, validator, hyper_gradient_dir):
        self.training_dataset = training_dataset
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
        return HyperGradientTrainer.__create_gradient_matrix(
            cache_size, mask, gradient_shape, hyper_gradient_dir
        )
