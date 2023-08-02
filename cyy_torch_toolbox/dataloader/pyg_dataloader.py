from typing import Any

import torch
import torch.utils.data
import torch_geometric

from ..dataset_util import GraphDatasetUtil


class RandomNodeLoader(torch.utils.data.DataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.RandomNodeLoader`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """

    def __init__(
        self,
        dataset_util: GraphDatasetUtil,
        **kwargs: Any,
    ) -> None:
        assert len(dataset_util.dataset) == 1
        self.dataset_util = dataset_util
        self.node_indices = torch_geometric.utils.mask_to_index(
            self.dataset_util.dataset[0]["mask"]
        ).tolist()

        assert "collate_fn" not in kwargs
        super().__init__(
            self.node_indices,
            collate_fn=self.__collate_fn,
            **kwargs,
        )

    def __collate_fn(self, indices):
        batch = self.dataset_util.dataset[0] | {
            "batch_node_indices": indices,
            "batch_size": len(indices),
        }
        batch["inputs"] = batch["input"]
        batch["targets"] = batch["target"]
        return batch
