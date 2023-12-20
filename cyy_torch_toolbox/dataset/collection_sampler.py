import functools
from typing import Any

from ..factory import Factory
from ..ml_type import MachineLearningPhase, TransformType
from .classification_collection import ClassificationDatasetCollection
from .collection import DatasetCollection
from .sampler import DatasetSampler
from .util import DatasetUtil


class DatasetCollectionSampler:
    def __init__(
        self,
        dataset_collection: DatasetCollection | ClassificationDatasetCollection,
        part_number: int,
    ) -> None:
        self._dataset_indices: dict[MachineLearningPhase, dict] = {}
        self._dc = dataset_collection
        self._part_number = part_number
        self._samplers: dict[MachineLearningPhase, DatasetSampler] = {}
        self.set_dataset_collection(dataset_collection)

    def set_dataset_collection(
        self, dataset_collection: DatasetCollection | ClassificationDatasetCollection
    ) -> None:
        self._dc = dataset_collection
        self._samplers = {
            phase: DatasetSampler(dataset_collection.get_dataset_util(phase))
            for phase in MachineLearningPhase
        }

    def __getstate__(self) -> dict:
        # capture what is normally pickled
        state = self.__dict__.copy()
        state["_samplers"] = None
        state["_dc"] = None
        return state

    def sample(self, part_id: int) -> None:
        for phase in MachineLearningPhase:
            indices = self._dataset_indices[phase][part_id]
            assert indices
            self._dc.set_subset(phase=phase, indices=indices)


class IIDSampler(DatasetCollectionSampler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        parts: list[float] = [1] * self._part_number
        for phase in MachineLearningPhase:
            self._dataset_indices[phase] = dict(
                enumerate(self._samplers[phase].iid_split_indices(parts))
            )


class IIDFlipSampler(IIDSampler):
    def __init__(
        self,
        dataset_collection: ClassificationDatasetCollection,
        part_number: int,
        flip_percent: float | list[dict[Any, float]],
    ) -> None:
        super().__init__(dataset_collection=dataset_collection, part_number=part_number)
        self._flipped_indices: dict = {}
        for phase, part_indices in self._dataset_indices.items():
            if phase != MachineLearningPhase.Training:
                continue
            if isinstance(flip_percent, (dict, list)):
                for part_index, indices in part_indices.items():
                    self._flipped_indices |= self._samplers[
                        phase
                    ].randomize_label_by_class(
                        checked_indices=indices,
                        percent=flip_percent[part_index],
                        all_labels=dataset_collection.get_labels(),
                    )
            else:
                assert isinstance(flip_percent, float)
                for indices in part_indices.values():
                    self._flipped_indices |= self._samplers[phase].randomize_label(
                        indices=indices,
                        percent=flip_percent,
                        all_labels=dataset_collection.get_labels(),
                    )
        assert self._flipped_indices

    @classmethod
    def __transform_target(cls, flipped_indices, target, index):
        if index in flipped_indices:
            return DatasetUtil.replace_target(target, {target: flipped_indices[index]})
        return target

    def sample(self, part_id: int) -> None:
        super().sample(part_id=part_id)
        for phase in MachineLearningPhase:
            if phase != MachineLearningPhase.Training:
                continue
            indices = self._dataset_indices[phase][part_id]
            assert indices
            index_list = sorted(indices)
            new_flipped_dict = {}
            for new_idx, idx in enumerate(sorted(index_list)):
                if idx in self._flipped_indices:
                    new_flipped_dict[new_idx] = self._flipped_indices[idx]
            assert new_flipped_dict
            self._dc.append_transform(
                transform=functools.partial(self.__transform_target, new_flipped_dict),
                key=TransformType.Target,
                phases=[phase],
            )


class RandomSampler(DatasetCollectionSampler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        parts: list[float] = [1] * self._part_number
        for phase in MachineLearningPhase:
            self._dataset_indices[phase] = dict(
                enumerate(self._samplers[phase].random_split_indices(parts))
            )


global_sampler_factory = Factory()
global_sampler_factory.register("iid", IIDSampler)
global_sampler_factory.register("iid_flip", IIDFlipSampler)
global_sampler_factory.register("random", RandomSampler)


def get_dataset_collection_sampler(
    name: str, dataset_collection: ClassificationDatasetCollection, **kwargs
) -> DatasetCollectionSampler:
    constructor = global_sampler_factory.get(name.lower())
    if constructor is None:
        raise NotImplementedError(name)
    return constructor(dataset_collection=dataset_collection, **kwargs)
