from typing import Callable, Any
from shapely_value import ShapleyValue


def create_shapley_value_algorithm(algorithm_name: str, **kwargs):
    classes = {"g_shapley_value": ShapleyValue}
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
