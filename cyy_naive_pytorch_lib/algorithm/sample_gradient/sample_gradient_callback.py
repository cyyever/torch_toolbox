from callback import Callback

from .sample_gradient import get_sample_gradient


class SampleGradientCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__sample_gradients = dict()
        self.__computed_indices = None

    @property
    def sample_gradients(self):
        return self.__sample_gradients

    def set_computed_indices(self, computed_indices):
        self.__computed_indices = set(computed_indices)

    def _before_batch(self, **kwargs):
        trainer = kwargs["model_executor"]
        batch = kwargs["batch"]

        instance_inputs, instance_targets, instance_info = trainer.decode_batch(batch)
        assert "index" in instance_info
        instance_indices = instance_info["index"]
        instance_indices = {idx.data.item() for idx in instance_indices}
        batch_gradient_indices: set = instance_indices
        if self.__computed_indices is not None:
            batch_gradient_indices &= self.__computed_indices
        self.__sample_gradients.clear()
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
        gradient_list = get_sample_gradient(
            trainer.model_with_loss,
            sample_gradient_inputs,
            sample_gradient_targets,
        )

        assert len(gradient_list) == len(sample_gradient_indices)
        for (sample_gradient, index) in zip(gradient_list, sample_gradient_indices):
            self.__sample_gradients[index] = sample_gradient

    def _after_execute(self, **kwargs):
        self.__sample_gradients.clear()
