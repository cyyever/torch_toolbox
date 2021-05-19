from hook import Callback

from .gradient_sanitizer import GradientSanitizer


class TrainerDebugger(Callback):
    def append_to_model_executor(self, model_executor):
        gradient_sanitizer = GradientSanitizer()
        gradient_sanitizer.append_to_model_executor(model_executor)
