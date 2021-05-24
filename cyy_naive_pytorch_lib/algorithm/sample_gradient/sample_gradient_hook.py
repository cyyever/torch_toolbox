import copy

from data_structure.synced_tensor_dict import SyncedTensorDict
from hooks.add_index_to_dataset import AddIndexToDataset

from .sample_gradient import get_sample_gradient


class SampleGradientHook(AddIndexToDataset):
    def __init__(self, *args, **kwargs):
        storage_dir = kwargs.pop("storage_dir", None)
        super().__init__(*args, **kwargs)
        self.__computed_indices = None
        self.__sample_gradient_dict = SyncedTensorDict.create()
        self.__storage_dir = storage_dir
        self.__model_with_loss = None
        if storage_dir is not None:
            self.__sample_gradient_dict.set_storage_dir(storage_dir)

    @property
    def sample_gradient_dict(self):
        return self.__sample_gradient_dict

    def set_computed_indices(self, computed_indices):
        self.__computed_indices = set(computed_indices)

    def _before_batch(self, **kwargs):
        self.sample_gradient_dict.clear()

    def _after_optimizer_step(self, **kwargs):
        trainer = kwargs["model_executor"]
        batch = kwargs["batch"]

        instance_inputs, instance_targets, instance_info = trainer.decode_batch(batch)
        assert "index" in instance_info
        instance_indices = {idx.data.item() for idx in instance_info["index"]}
        batch_gradient_indices: set = instance_indices
        if self.__computed_indices is not None:
            batch_gradient_indices &= self.__computed_indices
        sample_gradient_inputs = []
        sample_gradient_targets = []
        sample_gradient_indices = []
        for (instance_input, instance_target, instance_index) in zip(
            instance_inputs, instance_targets, instance_indices
        ):
            if instance_index not in batch_gradient_indices:
                continue
            sample_gradient_inputs.append(instance_input)
            sample_gradient_targets.append(instance_target)
            sample_gradient_indices.append(instance_index)
        if not sample_gradient_indices:
            return
        if self.__model_with_loss is None:
            self.__model_with_loss = trainer.copy_model_with_loss(True)
        gradient_list = get_sample_gradient(
            self.__model_with_loss,
            sample_gradient_inputs,
            sample_gradient_targets,
        )

        assert len(gradient_list) == len(sample_gradient_indices)
        for (sample_gradient, index) in zip(gradient_list, sample_gradient_indices):
            self.sample_gradient_dict[index] = sample_gradient

    def _after_execute(self, **kwargs):
        if not self.__storage_dir:
            self.sample_gradient_dict.clear()
        super()._after_execute(**kwargs)
