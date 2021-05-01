from ml_type import ModelExecutorCallbackPoint


class Callback:
    def append_to_model_executor(self, model_executor):
        for cb_point, name, method in self.__yield_callbacks():
            model_executor.append_callback(cb_point, name, method)

    def prepend_to_model_executor(self, model_executor):
        for cb_point, name, method in self.__yield_callbacks():
            model_executor.prepend_callback(cb_point, name, method)

    def prepend_to_model_executor_before_other_callback(
        self, model_executor, other_callback
    ):
        for cb_point, name, method in self.__yield_callbacks():
            res = other_callback.__get_callback(cb_point)
            if res is None:
                model_executor.prepend_callback(cb_point, name, method)
            else:
                other_name = res[1]
                model_executor.prepend_before_other_callback(
                    cb_point, name, method, other_name
                )

    def set_stripable(self, model_executor):
        for name in self.__yield_callback_names():
            model_executor.set_stripable_callback(name)

    def remove_from_model_executor(self, model_executor):
        for name in self.__yield_callback_names():
            model_executor.remove_callback(name)

    def __yield_callback_names(self):
        for _, name, __ in self.__yield_callbacks():
            yield name

    def __get_callback(self, cb_point):
        method_name = "_" + str(cb_point).split(".")[-1].lower()
        name = self.__class__.__name__ + "." + str(method_name)
        if hasattr(self, method_name):
            return (cb_point, name, getattr(self, method_name))
        return None

    def __yield_callbacks(self):
        for cb_point in ModelExecutorCallbackPoint:
            res = self.__get_callback(cb_point)
            if res is not None:
                yield res
