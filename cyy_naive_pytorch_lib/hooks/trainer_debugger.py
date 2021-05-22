from hook import Callback

from .gradient_sanitizer import GradientSanitizer
from .memory_tracker import MemoryTracker


class TrainerDebugger(Callback):
    def append_to_model_executor(self, model_executor):
        GradientSanitizer().append_to_model_executor(model_executor)
        MemoryTracker().append_to_model_executor(model_executor)
