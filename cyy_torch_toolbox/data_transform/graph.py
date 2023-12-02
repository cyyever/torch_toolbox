from typing import Any


def pyg_data_extraction(data: Any) -> dict | None:
    if "input" in data:
        return data
    match data:
        case {
            "graph_index": graph_index,
            "original_dataset": original_dataset,
        }:
            graph = original_dataset[graph_index]
            return data | {
                "input": graph,
                "target": graph.y,
            }
    return None
