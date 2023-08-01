from typing import Any

import torch

from ..dataset_util import GraphDatasetUtil
from ..dependency import has_torch_geometric
from .base import ModelEvaluator

if has_torch_geometric:
    import torch_geometric.nn
    import torch_geometric.utils


class GraphModelEvaluator(ModelEvaluator):
    def __init__(self, edge_dict: dict, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.edge_dict: dict = edge_dict
        self.node_and_neighbour_index_map: dict = {}
        self.edge_index_map: dict = {}
        self.node_and_neighbour_mask: dict = {}
        self.neighbour_hop = sum(
            1
            for _, module in self.model_util.get_modules()
            if isinstance(module, torch_geometric.nn.MessagePassing)
        )

    def __call__(self, **kwargs: Any) -> dict:
        print(kwargs)
        inputs = kwargs["inputs"]
        mask = kwargs["mask"]
        phase = kwargs["phase"]
        if phase not in self.edge_index_map:
            edge_index = inputs["edge_index"]
            node_indices = set(torch_geometric.utils.mask_to_index(mask).tolist())
            node_and_neighbours: set = GraphDatasetUtil.get_neighbors(
                node_indices=node_indices,
                edge_dict=self.edge_dict,
                hop=self.neighbour_hop,
            )
            node_and_neighbour_index_map = {
                node_index: idx
                for idx, node_index in enumerate(sorted(node_and_neighbours))
            }
            self.node_and_neighbour_index_map[phase] = node_and_neighbour_index_map
            new_source_list = []
            new_target_list = []
            assert torch_geometric.utils.is_undirected(edge_index=edge_index)
            for source in node_and_neighbour_index_map:
                for target in self.edge_dict[source]:
                    if target >= source and target in node_and_neighbour_index_map:
                        new_source_list.append(node_and_neighbour_index_map[source])
                        new_target_list.append(node_and_neighbour_index_map[target])
            edge_index = torch.tensor(
                data=[new_source_list, new_target_list], dtype=edge_index.dtype
            )
            self.edge_index_map[phase] = edge_index
            node_and_neighbour_mask = torch.zeros_like(mask)
            for idx in node_and_neighbours:
                node_and_neighbour_mask[idx] = True
            self.node_and_neighbour_mask[phase] = node_and_neighbour_mask

        inputs["edge_index"] = self.edge_index_map[phase]
        print(inputs["x"].shape, mask.shape)
        inputs["x"] = inputs["x"][self.node_and_neighbour_mask[phase]]

        batch_mask = torch_geometric.utils.index_to_mask(
            torch.tensor(kwargs["batch_node_indices"]), kwargs["targets"].shape[0]
        )
        kwargs["targets"] = kwargs["targets"][batch_mask]
        batch_mask = torch.zeros(
            (len(self.node_and_neighbour_index_map[phase]),), dtype=torch.bool
        )
        for idx in kwargs["batch_node_indices"]:
            batch_mask[self.node_and_neighbour_index_map[phase][idx]] = True
        kwargs["mask"] = batch_mask
        return super().__call__(**kwargs)

    def _compute_loss(
        self, output: torch.Tensor, mask: torch.Tensor, **kwargs: Any
    ) -> dict:
        output = output[mask]
        return super()._compute_loss(output=output, **kwargs)
