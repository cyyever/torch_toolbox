from hook import ComposeHook, Hook

from .gradient_sanitizer import GradientSanitizer

# from .memory_tracker import MemoryTracker


class TrainerDebugger(ComposeHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gradient_sanitizer = GradientSanitizer()
