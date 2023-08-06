from typing import Any

import torch

from ..dataset_util import GraphDatasetUtil
from ..dependency import has_torch_geometric
from .base import ModelEvaluator

if has_torch_geometric:
    import torch_geometric.nn
    import torch_geometric.utils


class GraphModelEvaluator(ModelEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__edge_dict: dict = {}
        self.__neighbour_edge_dicts: dict = {}
        self.batch_neighbour_index_map: dict = {}
        self.edge_index_map: dict = {}
        self.batch_neighbour_mask: dict = {}
        self.neighbour_hop = sum(
            1
            for _, module in self.model_util.get_modules()
            if isinstance(module, torch_geometric.nn.MessagePassing)
        )

    def __call__(self, **kwargs: Any) -> dict:
        graph = kwargs["original_dataset"][kwargs["graph_index"]]
        x = kwargs["input"]["x"]
        mask = kwargs["mask"]
        phase = kwargs["phase"]
        # get_logger().error("old edge shape is %s", graph.edge_index.shape)
        # get_logger().error("shape1 is %s shape 2 %s", x.shape, mask.shape)

        self.__narrow_batch(
            mask=mask,
            phase=phase,
            batch_node_indices=kwargs["batch_node_indices"],
            graph_dict=kwargs,
            edge_dtype=graph.edge_index.dtype,
        )
        inputs = {
            "edge_index": self.edge_index_map[phase],
            "x": x[self.batch_neighbour_mask[phase]],
        }

        batch_mask = torch_geometric.utils.index_to_mask(
            torch.tensor(kwargs["batch_node_indices"]), kwargs["target"].shape[0]
        )
        kwargs["targets"] = kwargs["target"][batch_mask]
        batch_mask = torch.zeros(
            (len(self.batch_neighbour_index_map[phase]),), dtype=torch.bool
        )
        for idx in kwargs["batch_node_indices"]:
            batch_mask[self.batch_neighbour_index_map[phase][idx]] = True
        kwargs["batch_mask"] = batch_mask
        # get_logger().error(
        #     "batch size is %s edge shape is %s new x shape is %s",
        #     batch_mask.sum().item(),
        #     self.edge_index_map[phase].shape,
        #     inputs["x"].shape,
        # )
        return super().__call__(inputs=inputs, **kwargs)

    def _compute_loss(
        self, output: torch.Tensor, batch_mask: torch.Tensor, **kwargs: Any
    ) -> dict:
        return super()._compute_loss(output=output[batch_mask], **kwargs)

    def __narrow_graph(self, mask, phase, graph_dict) -> None:
        if phase in self.__neighbour_edge_dicts:
            return
        if not self.__edge_dict:
            self.__edge_dict = GraphDatasetUtil.get_edge_dict(graph_dict=graph_dict)
        node_indices = set(torch_geometric.utils.mask_to_index(mask).tolist())
        _, neighbour_edges = GraphDatasetUtil.get_neighbors(
            node_indices=node_indices,
            edge_dict=self.__edge_dict,
            hop=self.neighbour_hop,
        )
        self.__neighbour_edge_dicts[phase] = GraphDatasetUtil.edge_to_dict(
            neighbour_edges
        )

    def __narrow_batch(
        self, mask, phase, graph_dict, batch_node_indices, edge_dtype
    ) -> None:
        self.__narrow_graph(mask=mask, phase=phase, graph_dict=graph_dict)
        batch_neighbour, batch_neighbour_edges = GraphDatasetUtil.get_neighbors(
            node_indices=batch_node_indices,
            edge_dict=self.__neighbour_edge_dicts[phase],
            hop=self.neighbour_hop,
        )
        batch_neighbour_index_map = {
            node_index: idx for idx, node_index in enumerate(sorted(batch_neighbour))
        }
        self.batch_neighbour_index_map[phase] = batch_neighbour_index_map
        new_source_list = []
        new_target_list = []
        for source, target in batch_neighbour_edges:
            new_source_list.append(batch_neighbour_index_map[source])
            new_target_list.append(batch_neighbour_index_map[target])
        self.edge_index_map[phase] = torch.tensor(
            data=[new_source_list, new_target_list], dtype=edge_dtype
        )
        self.batch_neighbour_mask[phase] = torch_geometric.utils.index_to_mask(
            torch.tensor(list(batch_neighbour)), size=mask.shape[0]
        )
