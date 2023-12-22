import copy
import functools
import random
from typing import Any, Callable

from cyy_naive_lib.log import get_logger

from .util import DatasetUtil


class DatasetSampler:
    def __init__(self, dataset_util: DatasetUtil) -> None:
        self.__dataset_util: DatasetUtil = dataset_util
        self._excluded_indices: set = set()
        self._checked_indices: set | None = None

    def set_excluded_indices(self, excluded_indices):
        self._excluded_indices = excluded_indices

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

    def get_subsets(self, index_list: list) -> list:
        return [
            self.__dataset_util.get_subset(indices=indices) for indices in index_list
        ]

    def split_indices(
        self,
        parts: list[dict[Any, float]],
        labels: list | None = None,
    ) -> list[set]:
        assert parts

        sub_index_list: list[set] = [set()] * len(parts)

        def __split_per_label(label, indices):
            nonlocal parts
            label_part = [part.get(label, 0) for part in parts]
            if sum(label_part) == 0:
                return set()
            part_index_lists = self.__split_index_list(label_part, list(indices))
            for i, part_index_list in enumerate(part_index_lists):
                sub_index_list[i] = sub_index_list[i] | set(part_index_list)
            return indices

        self.__check_sample_by_label(
            callback=__split_per_label,
            labels=labels,
        )
        return sub_index_list

    def sample_indices(
        self,
        parts: list[dict[Any, float]],
        labels: list | None = None,
    ) -> list[set]:
        assert parts

        sub_index_list: list[set] = [set()] * len(parts)

        def __sample_per_label(label, indices):
            nonlocal parts
            label_part = [part.get(label, 0) for part in parts]
            if sum(label_part) == 0:
                return set()
            part_index_lists = self.__split_index_list(label_part, list(indices))
            sampled_indices = set()
            for i, part_index_list in enumerate(part_index_lists):
                part_index_list = set(part_index_list[: int(label_part * len(indices))])
                sampled_indices.update(part_index_list)
                sub_index_list[i] = sub_index_list[i] | part_index_list
            return sampled_indices

        self.__check_sample_by_label(
            callback=__sample_per_label,
            labels=labels,
        )
        return sub_index_list

    def iid_split_indices(
        self,
        parts: list[float],
        labels: list | None = None,
    ) -> list[set]:
        assert parts

        if labels is None:
            labels = list(self.label_sample_dict.keys())
        return self.split_indices(
            parts=[{label: part for label in labels} for part in parts],
            labels=labels,
        )

    def random_split_indices(
        self,
        parts: list[float],
        labels: list | None = None,
    ) -> list[list]:
        collected_indices = set()

        def __collect(label, indices):
            collected_indices.update(indices)
            return indices

        self.__check_sample_by_label(callback=__collect, labels=labels)
        return self.__split_index_list(parts, list(collected_indices))

    def iid_split(self, parts: list[float], labels: list | None = None) -> list:
        return self.get_subsets(self.iid_split_indices(parts, labels=labels))

    def iid_sample_indices(self, percent: float) -> set:
        labels = list(self.label_sample_dict.keys())
        return self.sample_indices(
            [{label: percent for label in labels}], labels=labels
        )[0]

    def randomize_label(
        self, indices: list, percent: float, all_labels: set | None = None
    ) -> dict[int, set]:
        randomized_label_map: dict[int, set] = {}
        if all_labels is None:
            all_labels = set(self.label_sample_dict.keys())

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
        self, percent: float | dict[Any, float], all_labels: set | None = None, **kwargs
    ) -> dict[int, set]:
        randomized_label_map: dict[int, set] = {}

        def __randomize(label: set, indices):
            nonlocal randomized_label_map
            nonlocal percent
            if not indices:
                return indices

            if isinstance(percent, dict):
                new_percent = percent[label]
            else:
                new_percent = percent
            assert isinstance(new_percent, float | int)

            randomized_label_map |= self.randomize_label(
                indices=indices,
                percent=new_percent,
                all_labels=all_labels,
            )

            return indices

        self.__check_sample_by_label(callback=__randomize, **kwargs)

        return randomized_label_map

    @classmethod
    def __split_index_list(cls, parts: list[float], index_list: list) -> list[list]:
        assert index_list
        if len(parts) == 1:
            assert parts[0] != 0
            return [index_list]
        random.shuffle(index_list)
        part_lens: list[int] = []
        first_assert = True
        index_num = len(index_list)

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
                part_indices.append(index_list[0:part_len])
                index_list = index_list[part_len:]
            else:
                part_indices.append([])
        return part_indices

    def __check_sample_by_label(
        self,
        callback: Callable,
        labels: list | None = None,
    ) -> None:
        excluded_indices = copy.copy(self._excluded_indices)

        if labels is None:
            labels = list(self.label_sample_dict.keys())
        for label in labels:
            indices = self.label_sample_dict[label] - excluded_indices
            if self._checked_indices is None:
                remaining_indices = set(indices)
            else:
                remaining_indices = set(indices).intersection(self._checked_indices)
            if remaining_indices:
                resulting_indices = callback(label=label, indices=remaining_indices)
                excluded_indices.update(resulting_indices)
