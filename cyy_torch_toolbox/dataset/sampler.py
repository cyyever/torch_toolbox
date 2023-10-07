import functools
import random

from cyy_naive_lib.log import get_logger

from .util import DatasetUtil


class DatasetSampler:
    def __init__(self, dataset_util: DatasetUtil):
        self.__dataset_util = dataset_util

    @functools.cached_property
    def sample_label_dict(self) -> dict[int, set]:
        return dict(self.__dataset_util.get_batch_labels())

    @functools.cached_property
    def label_sample_dict(self) -> dict:
        label_sample_dict: dict = {}
        for index, labels in self.sample_label_dict.items():
            for label in labels:
                if label not in label_sample_dict:
                    label_sample_dict[label] = [index]
                else:
                    label_sample_dict[label].append(index)
        return label_sample_dict

    def iid_split_indices(self, parts: list) -> list:
        return self.__get_split_indices(parts, iid=True)

    def random_split_indices(self, parts: list) -> list:
        return self.__get_split_indices(parts, iid=False)

    def iid_split(self, parts: list) -> list:
        return self.get_subsets(self.iid_split_indices(parts))

    def get_subsets(self, indices_list: list) -> list:
        return [
            self.__dataset_util.get_subset(indices=indices) for indices in indices_list
        ]

    def __get_split_indices(self, parts: list, iid: bool = True) -> list[list]:
        assert parts
        if len(parts) == 1:
            return [list(range(len(self.__dataset_util)))]

        def split_index_impl(indices_list: list) -> list[list]:
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

        if not iid:
            index_list = list(range(len(self.__dataset_util)))
            random.shuffle(index_list)
            return split_index_impl(index_list)

        sub_index_list: list[list] = [[]] * len(parts)
        assigned_indices: set = set()
        for v in self.label_sample_dict.values():
            indices: set = set(v) - assigned_indices
            assigned_indices |= indices
            index_list = list(indices)
            random.shuffle(index_list)
            part_index_list = split_index_impl(index_list)
            for i, part_index in enumerate(part_index_list):
                sub_index_list[i] = sub_index_list[i] + part_index
        return sub_index_list

    def __sample_indices(self, percents: dict) -> dict:
        sample_indices: dict = {}
        for label, indices in self.label_sample_dict.items():
            percent = percents[label]
            sample_size = int(len(indices) * percent)
            if sample_size == 0:
                sample_indices[label] = []
            else:
                sample_indices[label] = random.sample(indices, k=sample_size)
        return sample_indices

    def iid_sample_indices(self, percent: float) -> dict:
        return self.__sample_indices(
            {label: percent for label in self.label_sample_dict}
        )

    def randomize_label(self, percent: float) -> dict:
        sample_indices: list = sum(self.iid_sample_indices(percent).values(), [])
        labels: set = set(self.label_sample_dict.keys())
        randomized_label_map: dict = {}
        for index in sample_indices:
            other_labels = list(labels - self.sample_label_dict[index])
            randomized_label_map[index] = random.sample(
                other_labels, min(len(other_labels), len(self.sample_label_dict[index]))
            )
        return randomized_label_map
