from hook import Hook

from .gradient_sanitizer import GradientSanitizer
from .memory_tracker import MemoryTracker
#from cyy_naive_lib.log import get_logger


class TrainerDebugger(Hook):
    def append_to_model_executor(self, model_executor):
        GradientSanitizer().append_to_model_executor(model_executor)
        #MemoryTracker().append_to_model_executor(model_executor)
