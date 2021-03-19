from algorithm.per_sample_gradient import get_per_sample_gradient
from callback import Callback


class SampleGradientCallback(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_gradients = dict()
        self._computed_indices = None

    def _before_batch(self, **kwargs):
        trainer = kwargs["model_executor"]
        batch = kwargs["batch"]

        instance_inputs, instance_targets, instance_info = trainer.decode_batch(batch)
        instance_indices = instance_info["index"]
        assert instance_indices
        instance_indices = {idx.data.item() for idx in instance_indices}
        batch_gradient_indices: set = instance_indices
        if self._computed_indices is not None:
            batch_gradient_indices &= self._computed_indices
        self._sample_gradients.clear()
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
        gradient_list = get_per_sample_gradient(
            trainer.model_with_loss,
            sample_gradient_inputs,
            sample_gradient_targets,
        )

        assert len(gradient_list) == len(sample_gradient_indices)
        for (sample_gradient, index) in zip(gradient_list, sample_gradient_indices):
            self._sample_gradients[str(index)] = sample_gradient
