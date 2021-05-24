import copy
import tempfile

from .sample_gradient_hook import SampleGradientHook


def get_sample_gradient_dict(inferencer, computed_indices=None):
    tmp_inferencer = copy.deepcopy(inferencer)
    hook = SampleGradientHook(storage_dir=tempfile.gettempdir())
    if computed_indices is not None:
        hook.set_computed_indices(computed_indices)
    hook.append_to_model_executor(tmp_inferencer)
    tmp_inferencer.inference(use_grad=True)
    return hook.sample_gradient_dict
