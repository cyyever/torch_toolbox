import copy
import functools
import os
import random
from collections.abc import Iterable
from typing import Any, Generator, Type

import torch
import torch.utils
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.storage import get_cached_data

from cyy_torch_toolbox.dataset import get_dataset_size, select_item, subset_dp
from cyy_torch_toolbox.dataset_transform.transform import Transforms
from cyy_torch_toolbox.dependency import has_torch_geometric, has_torchvision
from cyy_torch_toolbox.ml_type import DatasetType, MachineLearningPhase

if has_torchvision:
    import PIL
    import torchvision
if has_torch_geometric:
    import torch_geometric.data
    import torch_geometric.utils


class DatasetUtil:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        name: None | str = None,
        transforms: Transforms | None = None,
        cache_dir: None | str = None,
    ) -> None:
        self.dataset: torch.utils.data.Dataset | list = dataset
        self.__len: None | int = None
        self._name: str = name if name else ""
        self.__transforms: Transforms | None = transforms
        self._cache_dir = cache_dir

    def __len__(self) -> int:
        if self.__len is None:
            self.__len = get_dataset_size(self.dataset)
        return self.__len

    def decompose(self) -> None | dict:
        return None

    def get_samples(self, indices: Iterable | None = None) -> Iterable:
        items = select_item(dataset=self.dataset, indices=indices, mask=self.get_mask())
        if self.__transforms is None:
            return items
        for idx, sample in items:
            sample = self.__transforms.extract_data(sample)
            yield idx, sample

    def get_mask(self) -> None | list:
        return None

    def get_sample(self, index: int) -> Any:
        for _, sample in self.get_samples(indices=[index]):
            return sample
        return None

    @classmethod
    def __decode_target(cls: Type, target: Any) -> set:
        match target:
            case int() | str():
                return {target}
            case list():
                return set(target)
            case torch.Tensor():
                if target.numel() == 1:
                    return {target.item()}
                if (target <= 1).all().item() and (target >= 0).all().item():
                    # one hot vector
                    return set(target.nonzero().view(-1).tolist())
                raise NotImplementedError(f"Unsupported target {target}")
            case dict():
                if "labels" in target:
                    return set(target["labels"].tolist())
                if all(isinstance(s, str) and s.isnumeric() for s in target):
                    return {int(s) for s in target}
        raise RuntimeError("can't extract labels from target: " + str(target))

    def _get_sample_input(self, index: int, apply_transform: bool = True) -> Any:
        sample = self.get_sample(index)
        sample_input = sample["input"]
        if apply_transform:
            assert self.__transforms is not None
            sample_input = self.__transforms.transform_input(
                sample_input, apply_random=False
            )
        return sample_input

    def get_batch_labels(self, indices: None | Iterable = None) -> Generator:
        for idx, sample in self.get_samples(indices):
            target = sample["target"]
            if self.__transforms is not None:
                target = self.__transforms.transform_target(target)
            yield idx, DatasetUtil.__decode_target(target)

    def get_sample_label(self, index):
        for _, labels in self.get_batch_labels(indices=[index]):
            assert len(labels) == 1
            return next(iter(labels))
        return None

    def get_labels(self) -> set:
        return set().union(*tuple(set(labels) for _, labels in self.get_batch_labels()))

    def get_original_dataset(self) -> torch.utils.data.Dataset | list:
        dataset = self.dataset
        if hasattr(dataset, "original_dataset"):
            dataset = dataset.original_dataset
        return dataset

    def get_label_names(self) -> dict:
        original_dataset = self.get_original_dataset()
        if (
            hasattr(original_dataset, "classes")
            and original_dataset.classes
            and isinstance(original_dataset.classes[0], str)
        ):
            return dict(enumerate(original_dataset.classes))

        def get_label_name(container: set, index: int) -> set:
            label = self.get_sample_label(index)
            if isinstance(label, str):
                container.add(label)
            return container

        label_names: set = functools.reduce(get_label_name, range(len(self)), set())
        if label_names:
            return dict(enumerate(sorted(label_names)))
        raise RuntimeError("no label names detected")


class DatasetSplitter(DatasetUtil):
    __sample_label_dict: None | dict = None
    __label_sample_dict: None | dict = None

    @property
    def sample_label_dict(self) -> dict[int, list]:
        if self.__sample_label_dict is not None:
            return self.__sample_label_dict
        self.__sample_label_dict = dict(self.get_batch_labels())
        return self.__sample_label_dict

    @property
    def label_sample_dict(self) -> dict:
        if self.__label_sample_dict is not None:
            return self.__label_sample_dict
        self.__label_sample_dict = {}
        for index, labels in self.sample_label_dict.items():
            for label in labels:
                if label not in self.__label_sample_dict:
                    self.__label_sample_dict[label] = [index]
                else:
                    self.__label_sample_dict[label].append(index)
        return self.__label_sample_dict

    def iid_split_indices(self, parts: list) -> list:
        return self.__get_split_indices(parts, iid=True)

    def random_split_indices(self, parts: list) -> list:
        return self.__get_split_indices(parts, iid=False)

    def iid_split(self, parts: list) -> list:
        return self.split_by_indices(self.iid_split_indices(parts))

    def get_subset(self, indices):
        return subset_dp(self.dataset, indices)

    def split_by_indices(self, indices_list: list) -> list:
        return [self.get_subset(indices) for indices in indices_list]

    def __get_split_indices(self, parts: list, iid: bool = True) -> list[list]:
        assert parts
        if len(parts) == 1:
            return [list(range(len(self)))]

        def split_index_impl(indices_list: list) -> list[list]:
            part_lens = []
            first_assert = False

            for part in parts:
                assert part > 0
                part_len = int(len(indices_list) * part / sum(parts))
                if part_len == 0 and first_assert:
                    first_assert = False
                    get_logger().warning(
                        "has zero part when splitting list, %s %s",
                        len(indices_list),
                        parts,
                    )
                part_lens.append(part_len)
            part_lens[-1] += len(indices_list) - sum(part_lens)
            part_indices = []
            for part_len in part_lens:
                if part_len != 0:
                    part_indices.append(indices_list[0:part_len])
                    indices_list = indices_list[part_len:]
                else:
                    part_indices.append([])
            return part_indices

        if not iid:
            index_list = list(range(len(self)))
            random.shuffle(index_list)
            return split_index_impl(index_list)

        sub_index_list: list[list] = [[]] * len(parts)
        for v in self.label_sample_dict.values():
            v = copy.deepcopy(v)
            random.shuffle(v)
            part_index_list = split_index_impl(v)
            for i, part_index in enumerate(part_index_list):
                sub_index_list[i] = sub_index_list[i] + part_index
        return sub_index_list

    def sample_by_labels(self, percents: list[float]) -> dict:
        sample_indices: dict = {}
        for idx, label in enumerate(sorted(self.label_sample_dict.keys())):
            v = self.label_sample_dict[label]
            sample_size = int(len(v) * percents[idx])
            if sample_size == 0:
                sample_indices[label] = []
            else:
                sample_indices[label] = random.sample(v, k=sample_size)
        return sample_indices

    def iid_sample(self, percentage: float) -> dict:
        return self.sample_by_labels([percentage] * len(self.label_sample_dict))

    def randomize_subset_label(self, percentage: float) -> dict:
        sample_indices = self.iid_sample(percentage)
        labels = self.get_labels()
        randomized_label_map = {}
        for label, indices in sample_indices.items():
            other_labels = list(set(labels) - {label})
            for index in indices:
                randomized_label_map[index] = random.choice(other_labels)
        return randomized_label_map


class VisionDatasetUtil(DatasetSplitter):
    @functools.cached_property
    def channel(self):
        x = self._get_sample_input(0)
        assert x.shape[0] <= 3
        return x.shape[0]

    def get_mean_and_std(self):
        if self._name.lower() == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            return (mean, std)
        mean = torch.zeros(self.channel)
        for index in range(len(self)):
            x = self._get_sample_input(index)
            for i in range(self.channel):
                mean[i] += x[i, :, :].mean()
        mean.div_(len(self))

        wh = None
        std = torch.zeros(self.channel)
        for index in range(len(self)):
            x = self._get_sample_input(index)
            if wh is None:
                wh = x.shape[1] * x.shape[2]
            for i in range(self.channel):
                std[i] += torch.sum((x[i, :, :] - mean[i].data.item()) ** 2) / wh
        std = std.div(len(self)).sqrt()
        return mean, std

    def save_sample_image(self, index: int, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sample_input = self._get_sample_input(index, apply_transform=False)
        match sample_input:
            case PIL.Image.Image():
                sample_input.save(path)
            case _:
                torchvision.utils.save_image(sample_input, path)

    def get_sample_raw_input(self, index: int) -> Any:
        return self.get_sample_image(index=index)

    @torch.no_grad()
    def get_sample_image(self, index: int) -> Any:
        tensor = self._get_sample_input(index, apply_transform=False)
        if isinstance(tensor, PIL.Image.Image):
            return tensor
        grid = torchvision.utils.make_grid(tensor)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        return PIL.Image.fromarray(ndarr)


class TextDatasetUtil(DatasetSplitter):
    @torch.no_grad()
    def get_sample_text(self, index: int) -> str:
        return self._get_sample_input(index, apply_transform=False)

    def get_sample_raw_input(self, index: int) -> Any:
        return self.get_sample_text(index=index)


class GraphDatasetUtil(DatasetSplitter):
    def get_mask(self) -> list[torch.Tensor]:
        assert len(self.dataset) == 1
        if hasattr(self.dataset[0], "mask") or "mask" in self.dataset[0]:
            return [dataset["mask"] for dataset in self.dataset]
        mask = torch.ones((self.dataset[0].x.shape[0],), dtype=torch.bool)
        return [mask]

    @classmethod
    def to_edge_list(cls, edge_index: torch.Tensor) -> list:
        if isinstance(edge_index, torch.Tensor):
            return edge_index.transpose(0, 1).numpy()
        return edge_index

    @classmethod
    def create_edge_index(cls, source_node_list, target_node_list) -> torch.Tensor:
        return torch.tensor(data=[source_node_list, target_node_list], dtype=torch.long)

    @classmethod
    def edge_to_dict(cls, edge_index: torch.Tensor | Iterable) -> dict[int, list]:
        assert isinstance(edge_index, torch.Tensor)
        res: dict = {}
        edge_index = torch_geometric.utils.coalesce(edge_index)
        neighbor_list = []
        edge_list = cls.to_edge_list(edge_index)
        last_a = edge_list[0][0]
        for edge in edge_list:
            a = edge[0]
            b = edge[1]
            if a == last_a:
                neighbor_list.append(b)
            elif a > last_a:
                res[last_a] = neighbor_list
                last_a = a
                neighbor_list = [b]
            else:
                raise RuntimeError()
        assert last_a is not None and last_a not in res
        assert neighbor_list
        res[last_a] = neighbor_list
        return res

    def get_edge_index(self, graph_index) -> torch.Tensor:
        return self.get_graph(graph_index).edge_index

    def get_graph(self, graph_index) -> Any:
        graph_dict = self.dataset[graph_index]
        original_dataset = graph_dict["original_dataset"]
        graph_index = graph_dict["graph_index"]
        graph = original_dataset[graph_index]
        return graph

    def get_edge_dict(self, graph_index=0) -> dict:
        graph_dict = self.dataset[graph_index]
        edge_dict = graph_dict.get("edge_dict", None)
        assert not self.get_graph(graph_index).is_directed()
        if edge_dict is not None:
            get_logger().info("use custom edge dict")
            return edge_dict
        original_dataset = graph_dict["original_dataset"]
        graph_index = graph_dict["graph_index"]
        key: str = f"torch_toolbox_{self._name}_edge_dict_{graph_index}"
        if hasattr(original_dataset, key):
            get_logger().warning(
                "get cached edge_dict from graph %s %s",
                id(original_dataset),
                graph_index,
            )
            return getattr(original_dataset, key)

        def compute_fun():
            return self.edge_to_dict(
                edge_index=self.get_edge_index(graph_index=graph_index)
            )

        if self._cache_dir:
            edge_dict = get_cached_data(os.path.join(self._cache_dir, key), compute_fun)
        else:
            edge_dict = compute_fun()
        setattr(original_dataset, key, edge_dict)

        return edge_dict

    def get_boundary(self, node_indices: Iterable) -> dict:
        assert len(self.dataset) == 1
        res: dict = {}
        edge_dict = self.get_edge_dict(graph_index=0)[0]
        node_indices = set(node_indices)
        for node_idx in node_indices:
            boundary = edge_dict[node_idx] - node_indices
            if boundary:
                res[node_idx] = boundary
        return res

    @classmethod
    def get_neighbors(
        cls, node_indices: Iterable, edge_dict: dict, hop: int
    ) -> tuple[set, torch.Tensor]:
        assert hop > 0
        # old_neighbors: set = set(node_indices)
        neighbors: set = set(node_indices)
        source_node_list = []
        target_node_list = []
        unchecked_nodes: set = set()
        for i in range(hop):
            old_neighbors = copy.copy(neighbors)
            if i == 0:
                unchecked_nodes = old_neighbors
            for node in unchecked_nodes:
                edge_neighbor = edge_dict.get(node, None)
                if edge_neighbor is None:
                    continue
                source_node_list += [node] * len(edge_neighbor)
                target_node_list += edge_neighbor
                target_node_list += [node] * len(edge_neighbor)
                source_node_list += edge_neighbor
                neighbors |= set(edge_neighbor)
            unchecked_nodes = neighbors - old_neighbors

        return neighbors, cls.create_edge_index(source_node_list, target_node_list)

    def get_subset(self, indices: Iterable) -> list[dict]:
        return self.get_node_subset(indices)

    def get_node_subset(self, node_indices: Iterable | torch.Tensor) -> list[dict]:
        assert node_indices
        node_indices = torch.tensor(list(node_indices))
        result = []
        for idx, graph in enumerate(self.dataset):
            if isinstance(graph, dict):
                tmp = graph.copy()
                graph = self.get_graph(idx)
            else:
                tmp = {
                    "graph_index": idx,
                    "original_dataset": self.dataset,
                }
            tmp["mask"] = torch_geometric.utils.index_to_mask(
                node_indices, size=graph.x.shape[0]
            )
            result.append(tmp)
        return result

    def get_edge_subset(self, graph_idx: int, edge_dict: dict) -> list[dict]:
        dataset = copy.copy(self.dataset)
        dataset[graph_idx]["edge_dict"] = edge_dict
        return dataset

    def decompose(self) -> None | dict:
        mapping: dict = {
            MachineLearningPhase.Training: "train_mask",
            MachineLearningPhase.Validation: "val_mask",
            MachineLearningPhase.Test: "test_mask",
        }
        if not all(
            hasattr(self.dataset[0], mask_name) for mask_name in mapping.values()
        ):
            return None
        datasets: dict = {}
        for idx, graph in enumerate(self.dataset):
            for phase, mask_name in mapping.items():
                if phase not in datasets:
                    datasets[phase] = []
                datasets[phase].append(
                    {
                        "mask": getattr(graph, mask_name),
                        "graph_index": idx,
                        "original_dataset": self.dataset,
                    }
                )
        return datasets

    def get_original_dataset(self) -> torch.utils.data.Dataset:
        if "original_dataset" in self.dataset[0]:
            return self.dataset[0]["original_dataset"]
        return super().get_original_dataset()


def get_dataset_util_cls(dataset_type: DatasetType) -> Type:
    class_name: Type = DatasetSplitter
    match dataset_type:
        case DatasetType.Vision:
            class_name = VisionDatasetUtil
        case DatasetType.Text:
            class_name = TextDatasetUtil
        case DatasetType.Graph:
            class_name = GraphDatasetUtil
    return class_name
