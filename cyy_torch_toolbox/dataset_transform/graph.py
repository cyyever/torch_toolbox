from typing import Any


def pyg_data_extraction(data: Any, extract_index: bool = True) -> dict | None:
    match data:
        case {"mask": mask, "graph": graph, "original_dataset": _}:
            return {
                "input": {"x": graph.x, "edge_index": graph.edge_index},
                "target": graph.y,
                "mask": mask,
            }
        case {"input": {"x": _, "edge_index": __}, "target": ___, "mask": ____}:
            return data
    return None
