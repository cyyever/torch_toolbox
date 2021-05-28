import tempfile

from algorithm.sample_gradient.sample_gradient_util import \
    get_sample_gradient_dict
from data_structure.synced_tensor_dict import SyncedTensorDict
from inference import Inferencer

from .hydra_hook import HyDRAHook


class HyDRAAnalyzer:
    def __init__(
        self,
        inferencer: Inferencer,
        hyper_gradient_dir,
        training_set_size,
        cache_size=1024,
    ):
        self.inferencer: Inferencer = inferencer
        self.hydra_gradient = HyDRAHook.create_hypergradient_dict(
            cache_size, self.inferencer.model, storage_dir=hyper_gradient_dir
        )
        self.cache_size = cache_size
        self.training_set_size = training_set_size

    def get_subset_contributions(
        self, training_subset_dict: dict, test_subset_dict: dict
    ) -> dict:
        hyper_gradient_sum_dict = HyDRAHook.create_hypergradient_dict(
            self.cache_size, self.inferencer.model
        )
        hyper_gradient_sum_dict.set_storage_dir(tempfile.gettempdir())

        for k, indices in training_subset_dict.items():
            hyper_gradient_sum = None
            for (_, hyper_gradient) in self.hydra_gradient.iterate(indices):
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
            training_subset_indices = self.hydra_gradient.keys()
        return self.get_subset_contributions(
            {idx: [idx] for idx in training_subset_indices}, test_subset_dict
        )

    def __get_test_gradient_dict(self, test_subset_dict: dict):
        computed_indices: list = sum(test_subset_dict.values(), [])
        sample_gradient_dict = get_sample_gradient_dict(
            self.inferencer, computed_indices
        )
        test_gredient_dict = SyncedTensorDict.create(key_type=str)
        test_gredient_dict.set_storage_dir(tempfile.gettempdir())
        for test_key, indices in test_subset_dict.items():
            for idx, sample_gradient in sample_gradient_dict.iterate(indices):
                assert idx in indices
                if test_key not in test_gredient_dict:
                    test_gredient_dict[test_key] = sample_gradient
                else:
                    test_gredient_dict[test_key] += sample_gradient
        sample_gradient_dict.clear()
        return test_gredient_dict
