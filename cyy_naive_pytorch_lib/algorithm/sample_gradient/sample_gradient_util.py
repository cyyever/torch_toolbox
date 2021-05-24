import copy
import tempfile

from .sample_gradient_hook import SampleGradientHook


def get_sample_gradient_dict(inferencer, computed_indices=None):
    tmp_inferencer = copy.deepcopy(inferencer)
    callback = SampleGradientHook(storage_dir=tempfile.gettempdir())
    if computed_indices is not None:
        callback.set_computed_indices(computed_indices)
    callback.append_to_model_executor(tmp_inferencer)
    tmp_inferencer.inference(use_grad=True)
    return callback.sample_gradient_dict
