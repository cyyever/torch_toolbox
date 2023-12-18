import copy
import functools
import random
from typing import Any, Callable, Iterable

from cyy_naive_lib.log import get_logger

from .util import DatasetUtil


class DatasetSampler:
    def __init__(self, dataset_util: DatasetUtil) -> None:
        self.__dataset_util: DatasetUtil = dataset_util

    @functools.cached_property
    def sample_label_dict(self) -> dict[int, set]:
        return dict(self.__dataset_util.get_batch_labels())

    @functools.cached_property
    def label_sample_dict(self) -> dict[Any, set]:
        label_sample_dict: dict = {}
        for index, labels in self.sample_label_dict.items():
            for label in labels:
                if label not in label_sample_dict:
                    label_sample_dict[label] = {index}
                else:
                    label_sample_dict[label].add(index)
        return label_sample_dict

    @functools.cached_property
    def all_labels(self) -> set[set]:
        return set().union(
            *tuple(set(labels) for labels in self.sample_label_dict.values())
        )

    def __get_indices_by_label(
        self,
        labels: list | None = None,
        excluded_indices: Iterable[int] | None = None,
    ) -> dict[Any, set]:
        if labels is None:
            labels = list(self.label_sample_dict.keys())
        excluded_index_set = set()
        if excluded_indices:
            excluded_index_set = set(excluded_indices)
        return {
            label: self.label_sample_dict[label] - excluded_index_set
            for label in labels
        }

    def iid_split_indices(
        self,
        parts: list[float],
        labels: list | None = None,
        excluded_indices: Iterable[int] | None = None,
    ) -> list[set]:
        assert parts

        sub_index_list: list[set] = [set()] * len(parts)

        def __iid_spilit(label, indices):
            if not indices:
                return indices
            index_list = list(indices)
            random.shuffle(index_list)
            part_index_lists = self.__split_index_list(parts, index_list)
            for i, part_index_list in enumerate(part_index_lists):
                sub_index_list[i] = sub_index_list[i] | set(part_index_list)
            return indices

        self.__split_indices(
            callback=__iid_spilit, labels=labels, excluded_indices=excluded_indices
        )
        return sub_index_list

    def random_split_indices(
        self,
        parts: list[float],
        labels: list | None = None,
        excluded_indices: Iterable[int] | None = None,
    ) -> list[list]:
        collected_indices = set()

        def __collect(label, indices):
            collected_indices.update(indices)
            return indices

        self.__split_indices(
            callback=__collect, labels=labels, excluded_indices=excluded_indices
        )

        index_list = list(collected_indices)
        random.shuffle(index_list)
        return self.__split_index_list(parts, index_list)

    def iid_split(self, parts: list[float], labels: list | None = None) -> list:
        return self.get_subsets(self.iid_split_indices(parts, labels=labels))

    def get_subsets(self, indices_list: list) -> list:
        return [
            self.__dataset_util.get_subset(indices=indices) for indices in indices_list
        ]

    def iid_sample_indices(self, percent: float, **kwargs) -> set:
        sub_index_list: set = set()

        def __sample(label, indices):
            if not indices:
                return indices
            sample_size = int(len(indices) * percent)
            sub_index_list.update(random.sample(list(indices), k=sample_size))
            return indices

        self.__split_indices(callback=__sample, **kwargs)

        return sub_index_list

    def randomize_label(
        self, indices: list, percent: float, all_labels: set | None = None
    ) -> dict[int, set]:
        randomized_label_map: dict[int, set] = {}
        if all_labels is None:
            all_labels = self.all_labels

        flipped_indices = random.sample(list(indices), k=int(len(indices) * percent))
        for index in flipped_indices:
            other_labels = list(all_labels - self.sample_label_dict[index])
            randomized_label_map[index] = set(
                random.sample(
                    other_labels,
                    min(len(other_labels), len(self.sample_label_dict[index])),
                )
            )

        return randomized_label_map

    def randomize_label_by_class(
        self,
        percent: float | dict[Any, float],
        checked_indices: list | None = None,
        all_labels: set | None = None,
        **kwargs
    ) -> dict[int, set]:
        randomized_label_map: dict[int, set] = {}

        def __randomize(label: set, indices):
            nonlocal randomized_label_map
            nonlocal percent
            if not indices:
                return indices

            if isinstance(percent, dict):
                if len(label) == 1:
                    label = list[label][0]
                new_percent = percent[label]
            else:
                new_percent = percent
            assert isinstance(new_percent, float)

            randomized_label_map |= self.randomize_label(
                indices=checked_indices if checked_indices is not None else indices,
                percent=new_percent,
                all_labels=all_labels,
            )

            return indices

        self.__split_indices(callback=__randomize, **kwargs)

        return randomized_label_map

    @classmethod
    def __split_index_list(cls, parts: list[float], indices_list: list) -> list[list]:
        assert indices_list
        if len(parts) == 1:
            return [indices_list]
        part_lens: list[int] = []
        first_assert = True
        index_num = len(indices_list)

        for part in parts:
            assert part > 0
            part_len = int(index_num * part / sum(parts))
            if part_len == 0:
                if sum(part_lens, start=0) < index_num:
                    part_len = 1
                elif first_assert:
                    first_assert = False
                    get_logger().warning(
                        "has zero part when splitting list, %s %s",
                        index_num,
                        parts,
                    )
            part_lens.append(part_len)
        part_lens[-1] += index_num - sum(part_lens)
        part_indices = []
        for part_len in part_lens:
            if part_len != 0:
                part_indices.append(indices_list[0:part_len])
                indices_list = indices_list[part_len:]
            else:
                part_indices.append([])
        return part_indices

    def __split_indices(
        self,
        callback: Callable,
        labels: list | None = None,
        excluded_indices: Iterable[int] | None = None,
    ) -> None:
        if not excluded_indices:
            excluded_indices = set()
        else:
            excluded_indices = copy.copy(set(excluded_indices))

        label_sample_sub_dict: dict = self.__get_indices_by_label(
            labels=labels, excluded_indices=excluded_indices
        )
        for label, indices in label_sample_sub_dict.items():
            resulting_indices = callback(
                label=label, indices=set(indices) - excluded_indices
            )
            excluded_indices.update(resulting_indices)
