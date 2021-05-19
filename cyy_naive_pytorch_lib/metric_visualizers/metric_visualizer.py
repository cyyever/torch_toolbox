from hook import Callback


class MetricVisualizer(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__session_name = None

    def set_session_name(self, name: str):
        self.__session_name = name

    @property
    def session_name(self):
        return self.__session_name
