from typing import Any, Iterable

import torch
from cyy_naive_lib.log import get_logger

from ..dataset_util import GraphDatasetUtil
from ..dependency import has_torch_geometric
from ..ml_type import MachineLearningPhase
from .base import ModelEvaluator

if has_torch_geometric:
    import torch_geometric.nn
    import torch_geometric.utils


class GraphModelEvaluator(ModelEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.neighbour_hop = sum(
            1
            for _, module in self.model_util.get_modules()
            if isinstance(module, torch_geometric.nn.MessagePassing)
        )
        self.batch_neighbour_index_map: dict = {}
        self.__subset_edge_dict: dict = {}
        self.__batch_neighbour_edge_index: dict = {}
        get_logger().info("use neighbour_hop %s", self.neighbour_hop)

    def __call__(self, **kwargs: Any) -> dict:
        x = kwargs["input"]["x"]
        mask = kwargs["mask"]
        phase = kwargs["phase"]
        # get_logger().error("old edge shape is %s", graph.edge_index.shape)
        # get_logger().error("shape1 is %s shape 2 %s", x.shape, mask.shape)

        self.__narrow_graph(mask=mask, phase=phase, graph_dict=kwargs)
        batch_neighbour_mask = self.__narrow_batch(
            phase=phase,
            batch_node_indices=kwargs["batch_node_indices"],
            graph_dict=kwargs,
        )
        inputs = {
            "edge_index": self.__batch_neighbour_edge_index[phase],
            "x": x[batch_neighbour_mask],
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
        #     "batch size is %s %s edge shape is %s new x shape is %s",
        #     batch_mask.sum().item(),
        #     len(kwargs["batch_node_indices"]),
        #     self.edge_index_map[phase].shape,
        #     inputs["x"].shape,
        # )
        return super().__call__(inputs=inputs, **kwargs)

    def _compute_loss(
        self, output: torch.Tensor, batch_mask: torch.Tensor, **kwargs: Any
    ) -> dict:
        return super()._compute_loss(output=output[batch_mask], **kwargs)

    def __narrow_graph(
        self, mask: torch.Tensor, phase: MachineLearningPhase, graph_dict: dict
    ) -> None:
        if phase in self.__subset_edge_dict:
            return
        dataset_util = GraphDatasetUtil(dataset=[graph_dict])
        edge_dict = dataset_util.get_edge_dict(0)
        _, neighbour_edges = GraphDatasetUtil.get_neighbors(
            node_indices=torch_geometric.utils.mask_to_index(mask).tolist(),
            edge_dict=edge_dict,
            hop=self.neighbour_hop,
        )
        self.__subset_edge_dict[phase] = GraphDatasetUtil.edge_to_dict(neighbour_edges)

    def __narrow_batch(
        self,
        phase: MachineLearningPhase,
        batch_node_indices: Iterable,
        graph_dict: dict,
    ) -> torch.Tensor:
        graph = graph_dict["original_dataset"][graph_dict["graph_index"]]
        batch_neighbour, batch_neighbour_edges = GraphDatasetUtil.get_neighbors(
            node_indices=batch_node_indices,
            edge_dict=self.__subset_edge_dict[phase],
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
        self.__batch_neighbour_edge_index[phase] = torch.tensor(
            data=[new_source_list, new_target_list], dtype=graph.edge_index.dtype
        )
        return torch_geometric.utils.index_to_mask(
            torch.tensor(list(batch_neighbour)), size=graph_dict["mask"].shape[0]
        )
