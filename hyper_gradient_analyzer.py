import uuid
import copy
import os
import shutil
import torch
import torch.nn.utils.prune as prune
import cyy_pytorch_cpp
from cyy_naive_lib.log import get_logger

from .util import model_parameters_to_vector, get_model_sparsity, get_pruning_mask
from .hessian_vector_product import get_hessian_vector_product_func
from .hyper_gradient_trainer import HyperGradientTrainer


class hyper_gradientAnalyzer:
    def __init__(
            self,
            training_dataset,
            validator,
            hyper_gradient_dir,
            **kwargs):
        self.training_dataset = training_dataset
        self.validator = validator
        self.hyper_gradient_matrix = None

    def __load_hyper_gradients(self, hyper_gradient_dir):
        model = self.validator.model
        if prune.is_pruned(model):
            get_logger().info("use pruned model")
            parameters = model_parameters_to_vector(model)
            gradient_shape = parameters.shape
            mask = get_pruning_mask(model)
            assert len(mask) == len(parameters)
            self.hyper_gradient_matrix = HyperGradientTrainer.__create_gradient_matrix(
                100, mask, gradient_shape, hyper_gradient_dir)
        else:
            get_logger().info("use unpruned model")
            self.hyper_gradient_matrix = HyperGradientTrainer.__create_gradient_matrix(
                100, None, None, hyper_gradient_dir)
