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

    def __get_hook(self, cb_point):
        method_name = "_" + str(cb_point).split(".")[-1].lower()
        name = self.__class__.__name__ + "." + str(method_name)
        if hasattr(self, method_name):
            return (cb_point, name, getattr(self, method_name))
        return None

    def yield_hooks(self):
        for cb_point in ModelExecutorHookPoint:
            res = self.__get_hook(cb_point)
            if res is not None:
                yield res
