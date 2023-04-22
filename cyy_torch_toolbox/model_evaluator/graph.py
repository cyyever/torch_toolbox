import torch

from ..dataset_util import GraphDatasetUtil
from ..dependency import has_torch_geometric
from .base import ModelEvaluator

if has_torch_geometric:
    import torch_geometric.nn
    import torch_geometric.utils


class GraphModelEvaluator(ModelEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_and_neighbour_index_map = {}
        self.node_index_map = {}
        self.edge_index_map = {}
        self.neighbour_hop = sum(
            1
            for _, module in self.model_util.get_modules()
            if isinstance(module, torch_geometric.nn.MessagePassing)
        )

    def __call__(self, **kwargs) -> dict:
        inputs = kwargs["inputs"]
        inputs["x"] = inputs["x"][0]
        inputs["edge_index"] = inputs["edge_index"][0]
        mask = kwargs.pop("mask", None)
        if mask is None:
            return super().__call__(**kwargs)
        phase = kwargs["phase"]
        mask = mask.view(-1)
        if phase not in self.node_index_map:
            edge_index = inputs["edge_index"]
            node_indices = set(torch_geometric.utils.mask_to_index(mask).tolist())
            self.node_index_map[phase] = node_indices
            node_and_neighbours = GraphDatasetUtil.get_neighbors_from_edges(
                node_indices=node_indices, edge_index=edge_index, hop=self.neighbour_hop
            )
            tmp = set(node_and_neighbours.keys())
            for value in node_and_neighbours.values():
                tmp |= value
            node_and_neighbours = tmp
            node_and_neighbour_index_map = {
                node_index: idx
                for idx, node_index in enumerate(sorted(node_and_neighbours))
            }
            self.node_and_neighbour_index_map[phase] = node_and_neighbour_index_map
            new_source_list = []
            new_target_list = []
            for source, target in GraphDatasetUtil.foreach_edge(edge_index):
                if source in node_indices or target in node_indices:
                    new_source_list.append(node_and_neighbour_index_map[source])
                    new_target_list.append(node_and_neighbour_index_map[target])
            edge_index = torch.tensor(
                data=[new_source_list, new_target_list], dtype=edge_index.dtype
            )
            self.edge_index_map[phase] = edge_index
        else:
            node_indices = self.node_index_map[phase]
            node_and_neighbour_index_map = self.node_and_neighbour_index_map[phase]
            node_and_neighbours = set(node_and_neighbour_index_map.keys())
            edge_index = self.edge_index_map[phase]
        inputs["edge_index"] = edge_index
        kwargs["targets"] = kwargs["targets"][mask]
        new_mask = torch.zeros_like(mask)
        for idx in node_and_neighbours:
            new_mask[idx] = True
        inputs["x"] = inputs["x"][new_mask]
        new_mask = torch.zeros((len(node_and_neighbours),), dtype=torch.bool)
        for idx in node_indices:
            new_mask[node_and_neighbour_index_map[idx]] = True
        kwargs["mask"] = new_mask
        return super().__call__(**kwargs)

    def _forward_model(self, mask, **kwargs) -> dict | torch.Tensor:
        output = super()._forward_model(**kwargs)
        return output[mask]
