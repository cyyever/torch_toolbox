from typing import Any


def pyg_data_extraction(data: Any, extract_index: bool = True) -> dict | None:
    match data:
        case {
            "mask": mask,
            "graph_index": graph_index,
            "original_dataset": original_dataset,
        }:
            graph = original_dataset[graph_index]
            return {
                "input": {"x": graph.x, "edge_index": graph.edge_index},
                "target": graph.y,
                "mask": mask,
                "graph_index": graph_index,
                "original_dataset": original_dataset,
            }
        case {"input": {"x": _, "edge_index": __}, "target": ___, "mask": ____}:
            return data
    return None
