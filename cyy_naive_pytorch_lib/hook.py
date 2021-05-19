from ml_type import ModelExecutorHookPoint


class Callback:
    def append_to_model_executor(self, model_executor):
        for cb_point, name, method in self.__yield_hooks():
            model_executor.append_hook(cb_point, name, method)

    def prepend_to_model_executor(self, model_executor):
        for cb_point, name, method in self.__yield_hooks():
            model_executor.prepend_hook(cb_point, name, method)

    def prepend_to_model_executor_before_other_hook(
        self, model_executor, other_hook
    ):
        for cb_point, name, method in self.__yield_hooks():
            res = other_hook.__get_hook(cb_point)
            if res is None:
                model_executor.prepend_hook(cb_point, name, method)
            else:
                other_name = res[1]
                model_executor.prepend_before_other_hook(
                    cb_point, name, method, other_name
                )

    def set_stripable(self, model_executor):
        for name in self.__yield_hook_names():
            model_executor.set_stripable_hook(name)

    def remove_from_model_executor(self, model_executor):
        for name in self.__yield_hook_names():
            model_executor.remove_hook(name)

    def __yield_hook_names(self):
        for _, name, __ in self.__yield_hooks():
            yield name

    def __get_hook(self, cb_point):
        method_name = "_" + str(cb_point).split(".")[-1].lower()
        name = self.__class__.__name__ + "." + str(method_name)
        if hasattr(self, method_name):
            return (cb_point, name, getattr(self, method_name))
        return None

    def __yield_hooks(self):
        for cb_point in ModelExecutorHookPoint:
            res = self.__get_hook(cb_point)
            if res is not None:
                yield res
