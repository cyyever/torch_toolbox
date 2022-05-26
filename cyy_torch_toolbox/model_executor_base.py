import copy
from typing import Callable, Dict, List

from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import ModelExecutorHookPoint


class ModelExecutorBase:
    def __init__(self):
        self.__data: dict = {}
        self.__hooks: Dict[ModelExecutorHookPoint, List[Dict[str, Callable]]] = {}
        self.__stripable_hooks: set = set()
        self.__disabled_hooks: set = set()

    def get_data(self, key: str, default_value=None):
        return self.__data.get(key, default_value)

    def set_data(self, key: str, value) -> None:
        self.__data[key] = value

    def remove_data(self, key: str) -> None:
        self.__data.pop(key, None)

    def has_data(self, key: str) -> bool:
        return key in self.__data

    def clear_data(self):
        self.__data.clear()

    def exec_hooks(self, hook_point: ModelExecutorHookPoint, **kwargs):
        for hook in copy.copy(self.__hooks.get(hook_point, [])):
            for name, fun in copy.copy(hook).items():
                if name not in self.__disabled_hooks:
                    fun(model_executor=self, **kwargs)

    def has_hook(
        self,
        hook_point: ModelExecutorHookPoint,
    ):
        return hook_point in self.__hooks

    def hooks(self):
        return self.__hooks

    def disable_stripable_hooks(self):
        self.__disabled_hooks.update(self.__stripable_hooks)

    def enable_all_hooks(self):
        self.__disabled_hooks.clear()

    def append_named_hook(
        self,
        hook_point: ModelExecutorHookPoint,
        name: str,
        fun: Callable,
        stripable=False,
    ):
        self.insert_callback(-1, hook_point, name, fun, stripable)

    def prepend_named_hook(
        self,
        hook_point: ModelExecutorHookPoint,
        name: str,
        fun: Callable,
        stripable=False,
    ):
        self.insert_callback(0, hook_point, name, fun, stripable)

    def insert_callback(
        self,
        pos,
        hook_point: ModelExecutorHookPoint,
        name: str,
        fun: Callable,
        stripable=False,
    ):
        if stripable:
            self.__stripable_hooks.add(name)
        data = {name: fun}
        if hook_point not in self.__hooks:
            self.__hooks[hook_point] = [data]
        else:
            for d in self.__hooks[hook_point]:
                if name in d:
                    raise RuntimeError(name + " has registered")
            if pos < 0:
                self.__hooks[hook_point].append(data)
            else:
                self.__hooks[hook_point].insert(pos, data)

    def insert_hook(self, pos, hook: Hook):
        flag = False
        for hook_point, name, fun in hook.yield_hooks():
            self.insert_callback(pos, hook_point, name, fun, hook.stripable)
            flag = True
        assert flag

    def append_hook(self, hook: Hook):
        self.insert_hook(-1, hook)

    def prepend_hook(self, hook: Hook):
        self.insert_hook(0, hook)

    def enable_hook(self, hook: Hook):
        for name in hook.yield_hook_names():
            if name in self.__disabled_hooks:
                self.__disabled_hooks.remove(name)

    def disable_hook(self, hook: Hook):
        for name in hook.yield_hook_names():
            self.__disabled_hooks.add(name)

    def remove_hook_obj(self, hook: Hook):
        for hook_point, name, fun in hook.yield_hooks():
            self.remove_hook(name, hook_point)

    def remove_hook(self, name: str, hook_point: ModelExecutorHookPoint = None):
        for cur_hook_point, hooks in self.__hooks.items():
            if hook_point is not None and cur_hook_point != hook_point:
                continue
            for idx, hook in enumerate(hooks):
                hook.pop(name, None)
                hooks[idx] = hook
