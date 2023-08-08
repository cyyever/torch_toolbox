from typing import Any


def pyg_data_extraction(data: Any, extract_index: bool = True) -> dict | None:
    if "input" in data:
        return data
    match data:
        case {
            "graph_index": graph_index,
            "original_dataset": original_dataset,
        }:
            graph = original_dataset[graph_index]
            return data | {
                "input": {"x": graph.x, "edge_index": graph.edge_index},
                "target": graph.y,
            }
    return None
