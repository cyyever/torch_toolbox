from typing import Callable, Any


class ShapleyValue:
    def __init__(
        self,
        # algorithm_name: str,
        client_ids: set,
        round_id: int = None,
    ):
        # self.algorithm_name = algorithm_name
        self.client_ids = client_ids
        self.round_id = round_id
        # check algorithm name
        # raise xxxx

    def compute_shapely_value(
        self, information_callback: Callable[[int, set], dict]
    ) -> Any:
        raise NotImplementedError()


class GShapleyValue(ShapleyValue):
    def compute_shapely_value(
        self, information_callback: Callable[[int, set], dict]
    ) -> Any:

        # choose some clients

        # info = information_callback(clients)
        # info["accuracy"]
        # then compute
        return 0

