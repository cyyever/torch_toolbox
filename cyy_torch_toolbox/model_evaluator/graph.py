from cyy_naive_lib.log import get_logger
import torch

from ..dataset_util import GraphDatasetUtil
from ..dependency import has_torch_geometric
from .base import ModelEvaluator

if has_torch_geometric:
    import torch_geometric.nn
    import torch_geometric.utils


class GraphModelEvaluator(ModelEvaluator):
    def __init__(self, *args, **kwargs):
        edge_dict = kwargs.pop("edge_dict")
        super().__init__(*args, **kwargs)
        # self.node_index_map = {}
        self.node_and_neighbour_index_map = {}
        self.edge_index_map = {}
        self.node_and_neighbour_mask = {}
        self.node_mask = {}
        self.edge_dict = edge_dict
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
        if phase not in self.edge_index_map:
            edge_index = inputs["edge_index"]
            node_indices = set(torch_geometric.utils.mask_to_index(mask).tolist())
            # self.node_index_map[phase] = node_indices
            get_logger().error("id edge_dict %s",id(self.edge_dict))
            node_and_neighbours = GraphDatasetUtil.get_neighbors_from_edges(
                node_indices=node_indices,
                edge_dict=self.edge_dict,
                hop=self.neighbour_hop,
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
            node_and_neighbour_mask = torch.zeros_like(mask)
            for idx in node_and_neighbours:
                node_and_neighbour_mask[idx] = True
            self.node_and_neighbour_mask[phase] = node_and_neighbour_mask
            node_mask = torch.zeros((len(node_and_neighbours),), dtype=torch.bool)
            for idx in node_indices:
                node_mask[node_and_neighbour_index_map[idx]] = True
            self.node_mask[phase] = node_mask

        inputs["edge_index"] = self.edge_index_map[phase]
        kwargs["targets"] = kwargs["targets"][mask]
        inputs["x"] = inputs["x"][self.node_and_neighbour_mask[phase]]
        kwargs["mask"] = self.node_mask[phase]
        return super().__call__(**kwargs)

    def _forward_model(self, **kwargs) -> dict | torch.Tensor:
        mask = kwargs.pop("mask")
        output = super()._forward_model(**kwargs)
        return output[mask]
