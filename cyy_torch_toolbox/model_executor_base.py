from cyy_torch_toolbox.hook import HookCollection


class ModelExecutorBase(HookCollection):
    def __init__(self):
        super().__init__()
        self._data: dict = {}
        self._data["hooks"] = {}

    def exec_hooks(self, *args: list, **kwargs: dict) -> None:
        super().exec_hooks(*args, model_executor=self, **kwargs)
