import copy
from collections.abc import Iterable
from typing import Any, Callable, Dict, Generator, List

from ..ml_type import ExecutorHookPoint


class Hook:
    def __init__(self, stripable: bool = False) -> None:
        self.__stripable: bool = stripable
        self.is_cyy_torch_toolbox_hook: bool = True
        self._sub_hooks: list = []
        self._enabled: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            hasattr(value, "is_cyy_torch_toolbox_hook")
            and self.is_cyy_torch_toolbox_hook
        ):
            self._sub_hooks.append(value)
        super().__setattr__(name, value)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def stripable(self) -> bool:
        return self.__stripable

    def set_stripable(self) -> None:
        self.__stripable = True

    def yield_hook_names(self) -> Generator:
        for _, name, __ in self.yield_hooks():
            yield name

    def __get_hook(self, hook_point: ExecutorHookPoint) -> tuple | None:
        if not self._enabled:
            return None
        assert (
            str(hook_point).rsplit(".", maxsplit=1)[-1]
            == str(hook_point).split(".")[-1]
        )
        method_name = "_" + str(hook_point).rsplit(".", maxsplit=1)[-1].lower()
        name = self.__class__.__name__ + "." + str(method_name)
        if hasattr(self, method_name):
            return (hook_point, name, getattr(self, method_name))
        return None

    def yield_hooks(self) -> Generator:
        for c in self._sub_hooks:
            yield from c.yield_hooks()

        for hook_point in ExecutorHookPoint:
            res = self.__get_hook(hook_point)
            if res is not None:
                yield res


class HookCollection:
    def __init__(self) -> None:
        self.__hooks: Dict[ExecutorHookPoint, List[Dict[str, Callable]]] = {}
        self.__stripable_hooks: set = set()
        self.__disabled_hooks: set = set()
        self.__hook_objs: dict = {}

    def exec_hooks(self, hook_point: ExecutorHookPoint, **kwargs: Any) -> None:
        for hook in copy.copy(self.__hooks.get(hook_point, [])):
            for name, fun in copy.copy(hook).items():
                if name not in self.__disabled_hooks:
                    fun(**kwargs)

    def has_hook(
        self,
        hook_point: ExecutorHookPoint,
    ) -> bool:
        for hook in self.__hooks.get(hook_point, []):
            for name in hook:
                if name not in self.__disabled_hooks:
                    return True
        return False

    def hooks(self):
        return self.__hooks

    def disable_stripable_hooks(self) -> None:
        self.__disabled_hooks.update(self.__stripable_hooks)

    def enable_all_hooks(self) -> None:
        self.__disabled_hooks.clear()

    def append_named_hook(
        self,
        hook_point: ExecutorHookPoint,
        name: str,
        fun: Callable,
        stripable: bool = False,
    ) -> None:
        self.__insert_hook(-1, hook_point, name, fun, stripable)

    def prepend_named_hook(
        self,
        hook_point: ExecutorHookPoint,
        name: str,
        fun: Callable,
        stripable: bool = False,
    ) -> None:
        self.__insert_hook(0, hook_point, name, fun, stripable)

    def __insert_hook(
        self,
        pos: int,
        hook_point: ExecutorHookPoint,
        name: str,
        fun: Callable,
        stripable: bool = False,
    ) -> None:
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

    def insert_hook(self, pos: int, hook: Hook, hook_name: str | None = None) -> None:
        if hook_name is not None:
            assert hook_name not in self.__hook_objs
            self.__hook_objs[hook_name] = hook
        flag = False
        for hook_point, name, fun in hook.yield_hooks():
            self.__insert_hook(pos, hook_point, name, fun, hook.stripable)
            flag = True

        assert flag

    def get_hook(self, hook_name: str) -> Hook:
        return self.__hook_objs[hook_name]

    def get_hooks(self) -> Iterable:
        return self.__hook_objs.values()

    def has_hook_obj(self, hook_name: str) -> bool:
        return hook_name in self.__hook_objs

    def append_hook(self, hook: Hook, hook_name: str | None = None) -> None:
        self.insert_hook(-1, hook, hook_name)

    def prepend_hook(self, hook: Hook, hook_name: str | None = None) -> None:
        self.insert_hook(0, hook, hook_name)

    def enable_or_disable_hook(
        self, hook_name: str, enabled: bool, hook: Hook | None = None
    ) -> None:
        if enabled:
            if self.has_hook_obj(hook_name):
                self.enable_hook(hook_name=hook_name)
            else:
                assert hook is not None
                self.append_hook(hook, hook_name=hook_name)
        else:
            self.disable_hook(hook_name=hook_name)

    def enable_hook(self, hook_name: str, hook: Hook | None = None) -> None:
        if self.has_hook_obj(hook_name):
            hook = self.get_hook(hook_name)
        assert hook is not None
        for name in hook.yield_hook_names():
            if name in self.__disabled_hooks:
                self.__disabled_hooks.remove(name)

    def disable_hook(self, hook_name: str) -> None:
        if self.has_hook_obj(hook_name):
            hook = self.get_hook(hook_name)
            for name in hook.yield_hook_names():
                self.__disabled_hooks.add(name)

    def remove_hook(self, hook: Hook, hook_name: str | None = None) -> None:
        if hook_name is not None:
            hook = self.__hook_objs.pop(hook_name)
        for hook_point, name, _ in hook.yield_hooks():
            self.remove_named_hook(name, hook_point)

    def remove_named_hook(
        self, name: str, hook_point: ExecutorHookPoint | None = None
    ) -> None:
        for cur_hook_point, hooks in self.__hooks.items():
            if hook_point is not None and cur_hook_point != hook_point:
                continue
            for idx, hook in enumerate(hooks):
                hook.pop(name, None)
                hooks[idx] = hook
