from typing import Any, Iterable

import torch
from cyy_naive_lib.log import get_logger

from ..dataset_collection import DatasetCollection
from ..dataset_util import GraphDatasetUtil
from ..dependency import has_torch_geometric
from ..ml_type import MachineLearningPhase
from .base import ModelEvaluator

if has_torch_geometric:
    import torch_geometric.nn
    import torch_geometric.utils


class GraphModelEvaluator(ModelEvaluator):
    def __init__(self, dataset_collection: DatasetCollection, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__dc = dataset_collection
        self.neighbour_hop = sum(
            1
            for _, module in self.model_util.get_modules()
            if isinstance(module, torch_geometric.nn.MessagePassing)
        )
        self.batch_neighbour_index_map: dict = {}
        self.__subset_edge_dict: dict = {}
        self.__batch_neighbour_edge_index: dict = {}
        assert self.neighbour_hop == torch_geometric.utils.get_num_hops(self.model)
        get_logger().info("use neighbour_hop %s", self.neighbour_hop)

    def __call__(self, **kwargs: Any) -> dict:
        # kwargs["input"].x
        # mask = kwargs["mask"]
        phase = kwargs["phase"]
        graph_dict = self.__dc.get_dataset(phase=phase)[0]
        dataset_util = self.__dc.get_dataset_util(phase=phase)
        graph = self.__dc.get_dataset_util(phase=phase).get_graph(0)

        self.__narrow_graph(phase=phase, dataset_util=dataset_util)
        batch_neighbour_mask = self.__narrow_batch(
            phase=phase,
            batch_node_indices=kwargs["batch_node_indices"],
            graph_dict=graph_dict,
        )
        inputs = {
            "edge_index": self.__batch_neighbour_edge_index[phase],
            "x": graph.x[batch_neighbour_mask],
        }

        batch_mask = torch_geometric.utils.index_to_mask(
            torch.tensor(kwargs["batch_node_indices"]), graph.y.shape[0]
        )
        kwargs["targets"] = graph.y[batch_mask]
        batch_mask = torch.zeros(
            (len(self.batch_neighbour_index_map[phase]),), dtype=torch.bool
        )
        for idx in kwargs["batch_node_indices"]:
            batch_mask[self.batch_neighbour_index_map[phase][idx]] = True
        kwargs["batch_mask"] = batch_mask
        get_logger().debug(
            "batch size is %s edge shape is %s new x shape is %s",
            batch_mask.sum().item(),
            self.__batch_neighbour_edge_index[phase].shape,
            inputs["x"].shape,
        )
        return super().__call__(inputs=inputs, **kwargs)

    def _compute_loss(
        self, output: torch.Tensor, batch_mask: torch.Tensor, **kwargs: Any
    ) -> dict:
        return super()._compute_loss(output=output[batch_mask], **kwargs)

    def __narrow_graph(
        self, phase: MachineLearningPhase, dataset_util: GraphDatasetUtil
    ) -> None:
        if phase in self.__subset_edge_dict:
            return
        edge_dict = dataset_util.get_edge_dict()
        mask = dataset_util.get_mask()[0]
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
        batch_neighbour, batch_neighbour_edges = GraphDatasetUtil.get_neighbors(
            node_indices=batch_node_indices,
            edge_dict=self.__subset_edge_dict[phase],
            hop=self.neighbour_hop,
        )
        batch_neighbour = list(sorted(batch_neighbour))
        batch_neighbour_index_map = {
            node_index: idx for idx, node_index in enumerate(batch_neighbour)
        }
        self.batch_neighbour_index_map[phase] = batch_neighbour_index_map
        batch_neighbour_edges.apply_(lambda x: batch_neighbour_index_map[x])
        self.__batch_neighbour_edge_index[phase] = batch_neighbour_edges
        return torch_geometric.utils.index_to_mask(
            torch.tensor(batch_neighbour), size=graph_dict["mask"].shape[0]
        )
