from ml_type import ModelExecutorHookPoint


class Hook:
    def __init__(self, stripable=False):
        self.__stripable = stripable
        self._is_cyy_torch_toolbox_hook = True
        self._sub_hooks = []

    def __setattr__(self, name, value):
        if hasattr(value, "_is_cyy_torch_toolbox_hook"):
            self._sub_hooks.append(value)
        super().__setattr__(name, value)

    @property
    def stripable(self):
        return self.__stripable

    def set_stripable(self):
        self.__stripable = True

    def yield_hook_names(self):
        for _, name, __ in self.yield_hooks():
            yield name

    def _get_hook(self, cb_point):
        method_name = "_" + str(cb_point).rsplit(".", maxsplit=1)[0].lower()
        name = self.__class__.__name__ + "." + str(method_name)
        if hasattr(self, method_name):
            return (cb_point, name, getattr(self, method_name))
        return None

    def _get_hooks(self, cb_point):
        for c in self._sub_hooks:
            for hook in c._get_hooks(cb_point):
                yield hook
        res = self._get_hook(cb_point)
        if res is not None:
            yield res

    def yield_hooks(self):
        for cb_point in ModelExecutorHookPoint:
            for hook in self._get_hooks(cb_point):
                yield hook
