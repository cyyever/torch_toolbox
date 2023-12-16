import functools

from ..ml_type import MachineLearningPhase, TransformType
from .collection import DatasetCollection
from .sampler import DatasetSampler
from .util import DatasetUtil


class DatasetCollectionSampler:
    def __init__(self, dataset_collection: DatasetCollection) -> None:
        self._dataset_indices: dict[MachineLearningPhase, dict] = {}
        self._flipped_indices: dict = {}
        self._samplers: dict[MachineLearningPhase, DatasetSampler] = {
            phase: DatasetSampler(dataset_collection.get_dataset_util(phase))
            for phase in MachineLearningPhase
        }

    def __getstate__(self) -> dict:
        # capture what is normally pickled
        state = self.__dict__.copy()
        state["_samplers"] = None
        return state

    def sample(self, worker_id: int, dataset_collection: DatasetCollection) -> None:
        for phase in MachineLearningPhase:
            indices = self._dataset_indices[phase][worker_id]
            assert indices
            dataset_collection.set_subset(phase=phase, indices=indices)


class IIDSampler(DatasetCollectionSampler):
    def __init__(self, dataset_collection: DatasetCollection, part_number: int) -> None:
        super().__init__(dataset_collection=dataset_collection)
        parts: list[float] = [1] * part_number
        for phase in MachineLearningPhase:
            self._dataset_indices[phase] = dict(
                enumerate(self._samplers[phase].iid_split_indices(parts))
            )


class IIDFlipSampler(IIDSampler):
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        part_number: int,
        flip_percent: float,
    ) -> None:
        super().__init__(dataset_collection=dataset_collection, part_number=part_number)
        assert flip_percent > 0
        for phase, tmp in self._dataset_indices.items():
            if phase != MachineLearningPhase.Training:
                continue
            for indices in tmp.values():
                self._flipped_indices |= self._samplers[phase].randomize_label(
                    indices=indices, percent=flip_percent
                )
        assert self._flipped_indices

    @classmethod
    def __transform_target(cls, flipped_indices, target, index):
        if index in flipped_indices:
            return DatasetUtil.replace_target(target, flipped_indices[index])
        return target

    def sample(self, worker_id: int, dataset_collection: DatasetCollection) -> None:
        super().sample(worker_id=worker_id, dataset_collection=dataset_collection)
        for phase in MachineLearningPhase:
            indices = self._dataset_indices[phase][worker_id]
            assert indices
            index_list = sorted(indices)
            new_flipped_dict = {}
            for new_idx, idx in enumerate(sorted(index_list)):
                if idx in self._flipped_indices:
                    new_flipped_dict[new_idx] = self._flipped_indices[idx]
            assert new_flipped_dict
            dataset_collection.append_transform(
                transform=functools.partial(self.__transform_target, new_flipped_dict),
                key=TransformType.Target,
                phases=[phase],
            )


class RandomSampler(DatasetCollectionSampler):
    def __init__(self, dataset_collection: DatasetCollection, part_number: int) -> None:
        super().__init__(dataset_collection=dataset_collection)
        parts: list[float] = [1] * part_number
        for phase in MachineLearningPhase:
            self._dataset_indices[phase] = dict(
                enumerate(self._samplers[phase].random_split_indices(parts))
            )
