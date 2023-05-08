from typing import Any

import torch_geometric


def pyg_data_extraction(data: Any, extract_index: bool = True) -> dict | None:
    match data:
        case {
            "subset_mask": subset_mask,
            "graph": graph,
        }:
            res = {
                "input": {"x": graph.x, "edge_index": graph.edge_index},
                "target": graph.y,
                "mask": subset_mask,
            }
            return res
    return None
