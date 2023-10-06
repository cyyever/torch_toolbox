from typing import Any, Iterable

import torch
from cyy_naive_lib.log import get_logger

from ..dataset.util import GraphDatasetUtil
from ..dataset_collection import DatasetCollection
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
        self.neighbour_hop = torch_geometric.utils.get_num_hops(self.model)
        self.batch_neighbour_index_map: dict = {}
        self.__subset_edge_index: dict = {}
        self.__batch_neighbour_edge_index: dict = {}
        get_logger().debug("use neighbour_hop %s", self.neighbour_hop)

    def __call__(self, **kwargs: Any) -> dict:
        if "batch_node_indices" in kwargs:
            return self.__from_node_loader(**kwargs)
        return self.__from_neighbor_loader(**kwargs)

    def __from_neighbor_loader(self, **kwargs: Any) -> dict:
        inputs = {
            "edge_index": kwargs["edge_index"],
            "x": kwargs["x"],
            "n_id": kwargs["n_id"],
        }
        # batch_mask=self.tr
        mask = self.__dc.get_dataset_util(phase=kwargs["phase"]).get_mask()[0]
        batch_mask = mask[kwargs["n_id"]]
        kwargs["batch_size"]
        kwargs["targets"] = kwargs["y"][batch_mask]
        kwargs["batch_mask"] = batch_mask
        return super().__call__(inputs=inputs, **kwargs)

    def __from_node_loader(self, **kwargs: Any) -> dict:
        phase = kwargs["phase"]
        graph_dict = self.__dc.get_dataset(phase=phase)[0]
        graph = self.__dc.get_dataset_util(phase=phase).get_graph(0)

        dataset_util = self.__dc.get_dataset_util(phase=phase)
        self.__narrow_graph(phase=phase, dataset_util=dataset_util)
        batch_neighbour_mask, batch_neighbour_size = self.__narrow_batch(
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
        batch_mask = torch.zeros((batch_neighbour_size,), dtype=torch.bool)
        index_map = self.batch_neighbour_index_map[phase]
        for idx in kwargs["batch_node_indices"]:
            new_idx = index_map[idx]
            batch_mask[new_idx] = True
        kwargs["batch_mask"] = batch_mask
        return super().__call__(inputs=inputs, **kwargs)

    def _compute_loss(
        self, output: torch.Tensor, batch_mask: torch.Tensor, **kwargs: Any
    ) -> dict:
        if kwargs.pop("from_neighbor_loader", 0):
            pass
        return super()._compute_loss(output=output[batch_mask], **kwargs)

    def __narrow_graph(
        self, phase: MachineLearningPhase, dataset_util: GraphDatasetUtil
    ) -> None:
        if phase in self.__subset_edge_index:
            return
        mask = dataset_util.get_mask()[0]
        self.__subset_edge_index[phase] = torch_geometric.utils.k_hop_subgraph(
            node_idx=torch_geometric.utils.mask_to_index(mask).tolist(),
            num_hops=self.neighbour_hop,
            edge_index=dataset_util.get_edge_index(0),
            relabel_nodes=False,
            directed=True,
        )[1]

    def __narrow_batch(
        self,
        phase: MachineLearningPhase,
        batch_node_indices: Iterable,
        graph_dict: dict,
    ) -> tuple[torch.Tensor, int]:
        batch_node_indices = torch.tensor(sorted(batch_node_indices))
        (
            batch_neighbour,
            batch_neighbour_edge_index,
            _,
            __,
        ) = torch_geometric.utils.k_hop_subgraph(
            node_idx=batch_node_indices,
            edge_index=self.__subset_edge_index[phase],
            num_hops=self.neighbour_hop,
            relabel_nodes=True,
            directed=True,
        )
        batch_neighbour_index_map = {
            node_index.item(): idx for idx, node_index in enumerate(batch_neighbour)
        }
        self.batch_neighbour_index_map[phase] = batch_neighbour_index_map
        self.__batch_neighbour_edge_index[phase] = batch_neighbour_edge_index
        return torch_geometric.utils.index_to_mask(
            batch_neighbour, size=graph_dict["mask"].shape[0]
        ), len(batch_neighbour)
