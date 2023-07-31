from typing import Any, Union

import torch
import torch.utils.data
from torch import Tensor
from torch_geometric.data import Data, HeteroData


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
        data: Union[Data, HeteroData],
        **kwargs: Any,
    ) -> None:
        self.data = data
        assert isinstance(data, Data)
        self.num_nodes = data.num_nodes

        assert "collate_fn" not in kwargs
        super().__init__(
            list(range(self.num_nodes)),
            collate_fn=self.__collate_fn,
            **kwargs,
        )

    def __collate_fn(self, index):
        if not isinstance(index, Tensor):
            index = torch.tensor(index)

        return self.data.subgraph(index)
