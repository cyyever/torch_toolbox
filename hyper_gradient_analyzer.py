import copy
import tempfile

from cyy_naive_lib.log import get_logger

from .dataset import sub_dataset
from .hyper_gradient_trainer import HyperGradientTrainer
from .synced_tensor_dict_util import iterate_over_synced_tensor_dict


class HyperGradientAnalyzer:
    def __init__(self, validator, hyper_gradient_dir, cache_size=1024):
        assert validator.loss_fun.reduction in ("mean", "elementwise_mean")
        self.validator = validator

        self.cache_size = cache_size
        self.hyper_gradient_matrix = HyperGradientTrainer.create_gradient_matrix(
            self.cache_size, self.validator.model, hyper_gradient_dir)

    def get_contributions(self, training_set_size=None):
        if training_set_size is None:
            training_set_size = len(self.hyper_gradient_matrix)
        contribution_dict = dict()
        test_gradient = self.validator.get_gradient()

        for (sample_index, hyper_gradient) in iterate_over_synced_tensor_dict(
            self.hyper_gradient_matrix
        ):
            sample_index = int(sample_index)
            contribution_dict[sample_index] = (
                -(test_gradient @ hyper_gradient) / training_set_size
            ).data.item()

        assert len(contribution_dict) == training_set_size
        return contribution_dict

    def get_subset_contributions(
        self, training_subset_dict, test_subset_dict, training_set_size=None
    ):
        if training_set_size is None:
            training_set_size = len(self.hyper_gradient_matrix)
        hyper_gradient_sum_dict = HyperGradientTrainer.create_gradient_matrix(
            self.cache_size, self.validator.model
        )
        hyper_gradient_sum_dict.set_storage_dir(tempfile.gettempdir())

        for k, indices in training_subset_dict.items():
            hyper_gradient_sum = None
            for (_, hyper_gradient) in iterate_over_synced_tensor_dict(
                self.hyper_gradient_matrix, indices
            ):
                if hyper_gradient_sum is None:
                    hyper_gradient_sum = hyper_gradient
                else:
                    hyper_gradient_sum += hyper_gradient
            hyper_gradient_sum_dict[str(k)] = hyper_gradient_sum
        test_subset_gradient_dict = self.get_test_gradient_dict(
            test_subset_dict)
        contribution_dict = dict()
        for (training_key, hyper_gradient_sum) in iterate_over_synced_tensor_dict(
                hyper_gradient_sum_dict):
            training_key = int(training_key)
            contribution_dict[training_key] = dict()
            for (test_key, test_subset_gradient) in iterate_over_synced_tensor_dict(
                    test_subset_gradient_dict):
                contribution_dict[training_key][test_key] = (
                    -(test_subset_gradient @ hyper_gradient_sum) / training_set_size
                ).data.item()
        return contribution_dict

    def get_training_sample_contributions(
            self,
            test_subset_dict,
            training_subset_indices=None,
            training_set_size=None):
        if training_subset_indices is None:
            training_subset_indices = self.hyper_gradient_matrix.key()
        if training_set_size is None:
            training_set_size = len(self.hyper_gradient_matrix)
        contribution_dict = dict()

        test_subset_gradient_dict = self.get_test_gradient_dict(
            test_subset_dict)
        for (sample_index, hyper_gradient) in iterate_over_synced_tensor_dict(
            self.hyper_gradient_matrix, training_subset_indices
        ):
            get_logger().info("use sample %s", sample_index)
            sample_index = int(sample_index)
            contribution_dict[sample_index] = dict()
            for (test_key, test_subset_gradient) in iterate_over_synced_tensor_dict(
                    test_subset_gradient_dict):
                contribution_dict[sample_index][test_key] = (
                    -(test_subset_gradient @ hyper_gradient) / training_set_size
                ).data.item()
        return contribution_dict

    def get_test_gradients(self, test_subset_dict: dict):
        tmp_validator = copy.deepcopy(self.validator)
        for test_key, indices in test_subset_dict.items():
            subset = sub_dataset(self.validator.dataset, indices)
            assert len(subset) == len(indices)
            tmp_validator.set_dataset(subset)
            yield (test_key, tmp_validator.get_gradient() * len(subset))

    def get_test_gradient_dict(self, test_subset_dict: dict):
        test_gredient_dict = HyperGradientTrainer.create_gradient_matrix(
            self.cache_size
        )
        test_gredient_dict.set_storage_dir(tempfile.gettempdir())
        for (test_key, gradient) in self.get_test_gradients(test_subset_dict):
            assert test_key in test_gredient_dict
            test_gredient_dict[str(test_key)] = gradient
        return test_gredient_dict
