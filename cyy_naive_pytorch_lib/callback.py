from ml_type import ModelExecutorCallbackPoint


class Callback:
    def __init__(self):
        self.callback_names: list = []

    def append_to_model_executor(self, model_executor, stripable=False):
        for cb_point in ModelExecutorCallbackPoint:
            method_name = "_" + str(cb_point).split(".")[-1].lower()
            if hasattr(self, method_name):
                name = self.__class__.__name__ + "." + str(method_name)
                model_executor.add_named_callback(
                    cb_point, name, getattr(self, method_name), stripable=stripable
                )
                self.callback_names.append(name)

    def prepend_to_model_executor(self, model_executor, stripable=False):
        for cb_point in ModelExecutorCallbackPoint:
            method_name = "_" + str(cb_point).split(".")[-1].lower()
            if hasattr(self, method_name):
                name = self.__class__.__name__ + "." + str(method_name)
                model_executor.prepend_named_callback(
                    cb_point, name, getattr(self, method_name), stripable=stripable
                )
                self.callback_names.append(name)

    def remove_from_model_executor(self, model_executor):
        for name in self.callback_names:
            model_executor.remove_callback(name)
