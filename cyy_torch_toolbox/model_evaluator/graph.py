from typing import Any

import torch
from cyy_naive_lib.log import get_logger

from ..dataset_util import GraphDatasetUtil
from ..dependency import has_torch_geometric
from .base import ModelEvaluator

if has_torch_geometric:
    import torch_geometric.nn
    import torch_geometric.utils


class GraphModelEvaluator(ModelEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.edge_dict: dict = {}
        self.node_and_neighbour_index_map: dict = {}
        self.edge_index_map: dict = {}
        self.node_and_neighbour_mask: dict = {}
        self.neighbour_hop = sum(
            1
            for _, module in self.model_util.get_modules()
            if isinstance(module, torch_geometric.nn.MessagePassing)
        )

    def __call__(self, **kwargs: Any) -> dict:
        inputs = kwargs["inputs"]
        mask = kwargs["mask"]
        phase = kwargs["phase"]
        if not self.edge_dict:
            self.edge_dict = kwargs["edge_dict"]
        # get_logger().error("old edge shape is %s", inputs["edge_index"].shape)
        if phase not in self.edge_index_map:
            self.__narrow_graph(
                mask=mask, phase=phase, edge_dtype=inputs["edge_index"].dtype
            )
        masked = inputs["x"].shape[0] < mask.shape[0]
        if not masked:
            inputs["edge_index"] = self.edge_index_map[phase]
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
        kwargs["batch_mask"] = batch_mask
        get_logger().error(
            "batch size is %s edge shape is %s",
            batch_mask.sum().item(),
            self.edge_index_map[phase].shape,
        )
        return super().__call__(**kwargs)

    def _compute_loss(
        self, output: torch.Tensor, batch_mask: torch.Tensor, **kwargs: Any
    ) -> dict:
        output = output[batch_mask]
        return super()._compute_loss(output=output, **kwargs)

    def __narrow_graph(self, mask, phase, edge_dtype) -> None:
        if phase not in self.edge_index_map:
            node_indices = set(torch_geometric.utils.mask_to_index(mask).tolist())
            node_and_neighbours, neighbour_edges = GraphDatasetUtil.get_neighbors(
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
            for source, target in neighbour_edges:
                new_source_list.append(node_and_neighbour_index_map[source])
                new_target_list.append(node_and_neighbour_index_map[target])
            self.edge_index_map[phase] = torch.tensor(
                data=[new_source_list, new_target_list], dtype=edge_dtype
            )
            node_and_neighbour_mask = torch.zeros_like(mask)
            for idx in node_and_neighbours:
                node_and_neighbour_mask[idx] = True
            self.node_and_neighbour_mask[phase] = node_and_neighbour_mask
