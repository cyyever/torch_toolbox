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


def create_shapley_value_algorithm(algorithm_name: str, **kwargs):
    classes = {"g_shapley_value": GShapleyValue}
    if algorithm_name not in classes:
        raise RuntimeError("unknown name" + algorithm_name)
    return classes.get(algorithm_name, None)


if __name__ == "__main__":
    algorithm_constructor = create_shapley_value_algorithm(
        "g_shapley_value",
    )

    algorithm = algorithm_constructor(client_ids={1, 2, 3}, round_id=1)
    value = algorithm.compute_shapely_value(
        lambda client_id_set: {"accuracy": 0.9})
    print(value)
