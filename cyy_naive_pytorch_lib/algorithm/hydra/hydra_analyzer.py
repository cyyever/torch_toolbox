import copy
import tempfile

from dataset import sub_dataset
from inference import Inferencer

from .hydra_callback import HyDRACallback


class HyDRAAnalyzer:
    def __init__(
        self,
        inferencer: Inferencer,
        hyper_gradient_dir,
        training_set_size,
        cache_size=1024,
    ):
        self.inferencer: Inferencer = inferencer
        self.hyper_gradient_matrix = HyDRACallback.create_hypergradient_dict(
            cache_size, self.inferencer.model, storage_dir=hyper_gradient_dir
        )
        self.cache_size = cache_size
        self.training_set_size = training_set_size

    def get_subset_contributions(
        self, training_subset_dict: dict, test_subset_dict: dict
    ) -> dict:
        hyper_gradient_sum_dict = HyDRACallback.create_hypergradient_dict(
            self.cache_size, self.inferencer.model
        )
        hyper_gradient_sum_dict.set_storage_dir(tempfile.gettempdir())

        for k, indices in training_subset_dict.items():
            hyper_gradient_sum = None
            for (_, hyper_gradient) in self.hyper_gradient_matrix.iterate(indices):
                if hyper_gradient_sum is None:
                    hyper_gradient_sum = hyper_gradient
                else:
                    hyper_gradient_sum += hyper_gradient
            hyper_gradient_sum_dict[k] = hyper_gradient_sum
        test_subset_gradient_dict = self.__get_test_gradient_dict(test_subset_dict)
        contribution_dict: dict = dict()
        for (training_key, hyper_gradient_sum) in hyper_gradient_sum_dict.iterate():
            contribution_dict[training_key] = dict()
            for (test_key, test_subset_gradient) in test_subset_gradient_dict.iterate():
                contribution_dict[training_key][test_key] = (
                    -(test_subset_gradient @ hyper_gradient_sum)
                    / self.training_set_size
                ).data.item()
        return contribution_dict

    def get_training_sample_contributions(
        self, test_subset_dict, training_subset_indices=None
    ):
        if training_subset_indices is None:
            training_subset_indices = self.hyper_gradient_matrix.keys()
        return self.get_subset_contributions(
            {idx: [idx] for idx in training_subset_indices}, test_subset_dict
        )

    def __get_test_gradients(self, test_subset_dict: dict):
        tmp_inferencer = copy.deepcopy(self.inferencer)
        for test_key, indices in test_subset_dict.items():
            subset = sub_dataset(self.inferencer.dataset, indices)
            assert len(subset) == len(indices)
            tmp_inferencer.set_dataset(subset)
            yield (test_key, tmp_inferencer.get_gradient() * len(subset))

    def __get_test_gradient_dict(self, test_subset_dict: dict):
        test_gredient_dict = HyDRACallback.create_hypergradient_dict(self.cache_size)
        test_gredient_dict.set_storage_dir(tempfile.gettempdir())
        for (test_key, gradient) in self.__get_test_gradients(test_subset_dict):
            test_gredient_dict[test_key] = gradient
        return test_gredient_dict
