from ml_type import ModelExecutorHookPoint


class Hook:
    def __init__(self, stripable=False):
        self.__stripable = stripable

    @property
    def stripable(self):
        return self.__stripable

    def set_stripable(self):
        self.__stripable = True

    def yield_hook_names(self):
        for _, name, __ in self.yield_hooks():
            yield name

    def _get_hook(self, cb_point):
        method_name = "_" + str(cb_point).split(".")[-1].lower()
        name = self.__class__.__name__ + "." + str(method_name)
        if hasattr(self, method_name):
            return (cb_point, name, getattr(self, method_name))
        return None

    def yield_hooks(self):
        for cb_point in ModelExecutorHookPoint:
            res = self._get_hook(cb_point)
            if res is not None:
                yield res


class ComposeHook(Hook):
    def yield_hooks(self):
        components = [
            getattr(self, c) for c in dir(self) if isinstance(getattr(self, c), Hook)
        ]
        for cb_point in ModelExecutorHookPoint:
            for c in components:
                res = c._get_hook(cb_point)
                if res is not None:
                    yield res
            res = super()._get_hook(cb_point)
            if res is not None:
                yield res
