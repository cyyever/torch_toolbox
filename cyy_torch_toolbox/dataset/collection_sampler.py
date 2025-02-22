import copy
import functools
import json
import os
from dataclasses import dataclass

from cyy_naive_lib.log import log_info

from ..ml_type import Factory, MachineLearningPhase, TargetType, TransformType
from .classification_collection import ClassificationDatasetCollection
from .collection import DatasetCollection
from .sampler import DatasetSampler
from .util import DatasetUtil


@dataclass(kw_only=True)
class SampleInfo:
    indices: set[int] | list[int] | None = None
    file_path: str | None = None


class Base:
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        sample_phase: MachineLearningPhase | list[MachineLearningPhase] | None = None,
    ) -> None:
        self._dc = dataset_collection
        self._samplers: dict[MachineLearningPhase, DatasetSampler] = {}
        if isinstance(sample_phase, MachineLearningPhase):
            sample_phase = [sample_phase]
        self.sample_phase = sample_phase
        self.set_dataset_collection(dataset_collection)

    @property
    def dataset_collection(self) -> DatasetCollection:
        return self._dc

    def get_phases(self) -> list[MachineLearningPhase]:
        phases = []
        for phase in MachineLearningPhase:
            if self.sample_phase is not None and phase not in self.sample_phase:
                continue
            if not self._dc.has_dataset(phase):
                continue
            phases.append(phase)
        return phases

    def set_dataset_collection(self, dataset_collection: DatasetCollection) -> None:
        self._dc = dataset_collection
        self._samplers = {
            phase: DatasetSampler(dataset_collection.get_dataset_util(phase))
            for phase in self.get_phases()
        }

    def __getstate__(self) -> dict:
        # capture what is normally pickled
        state = self.__dict__.copy()
        state["_samplers"] = None
        return state


class SamplerBase(Base):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sample_info: dict[MachineLearningPhase, SampleInfo] = {}

    def sample(self) -> DatasetCollection:
        dc = copy.copy(self._dc)
        for phase in self.get_phases():
            indices = self._sample_info[phase].indices
            assert indices
            dc.set_subset(phase=phase, indices=set(indices))
        return dc

    def save(self, save_dir: str) -> None:
        with open(os.path.join(save_dir, "sampler.json"), "w", encoding="utf8") as f:
            json.dump(self._sample_info, f)


class SplitBase(Base):
    def __init__(self, part_number: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._part_number = part_number
        self._dataset_indices: dict[MachineLearningPhase, dict[int, SampleInfo]] = {}
        for phase in self.get_phases():
            for part_index in range(self._part_number):
                sample_info = self.get_preallocated_sample(
                    phase=phase, part_number=part_number, part_index=part_index
                )
                if sample_info is not None:
                    self._dataset_indices[phase][part_index] = sample_info
            if self._dataset_indices[phase]:
                assert len(self._dataset_indices[phase]) == part_number

    def get_preallocated_sample(
        self, phase: MachineLearningPhase, part_number: int, part_index: int
    ) -> None | SampleInfo:
        sampler = self._samplers[phase]
        original_dataset = getattr(sampler.dataset, "original_dataset")
        file_key = f"{str(phase).lower()}_files"
        if phase == MachineLearningPhase.Training:
            file_key = "train_files"
        files = getattr(original_dataset, file_key, [])
        print(type(original_dataset))
        assert isinstance(files, list)
        if len(files) == part_number:
            for file in files:
                if f"worker_{part_index}" in os.path.basename(file):
                    log_info("use path %s for index %s", file, part_index)
                    return SampleInfo(file_path=file)
        return None

    def set_split_indices(
        self, phase: MachineLearningPhase, index_result: dict[int, set[int] | list[int]]
    ) -> None:
        assert phase not in self._dataset_indices
        self._dataset_indices[phase] = {}
        for idx, dataset_indices in index_result.items():
            self._dataset_indices[phase][idx] = SampleInfo(indices=dataset_indices)

    def sample(self, part_index: int) -> DatasetCollection:
        dc = copy.copy(self._dc)
        for phase in self.get_phases():
            indices = self._dataset_indices[phase][part_index].indices
            assert indices
            dc.set_subset(phase=phase, indices=set(indices))
        return dc


class DatasetCollectionSplit(SplitBase):
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        part_proportions: list[dict[TargetType, float]],
    ) -> None:
        super().__init__(
            dataset_collection=dataset_collection, part_number=len(part_proportions)
        )
        for phase in self.get_phases():
            self.set_split_indices(
                phase=phase,
                index_result=dict(
                    enumerate(
                        self._samplers[phase].split_indices(
                            part_proportions=part_proportions
                        )
                    )
                ),
            )


class IIDSplit(SplitBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        parts: list[float] = [1] * self._part_number
        for phase in self.get_phases():
            self.set_split_indices(
                phase=phase,
                index_result=dict(
                    enumerate(self._samplers[phase].iid_split_indices(parts))
                ),
            )


class IIDSplitWithFlip(IIDSplit):
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        part_number: int,
        flip_percent: float
        | list[dict[TargetType, float]]
        | dict[int, dict[TargetType, float]],
    ) -> None:
        super().__init__(dataset_collection=dataset_collection, part_number=part_number)
        self.__flip_percent = flip_percent

    def get_flip_percent(self, part_index: int) -> float | dict:
        if isinstance(self.__flip_percent, dict | list):
            assert len(self.__flip_percent) == len(
                self._dataset_indices[MachineLearningPhase.Training]
            )
            assert len(self.__flip_percent) == len(
                self._dataset_indices[MachineLearningPhase.Validation]
            )
            return self.__flip_percent[part_index]
        assert isinstance(self.__flip_percent, float)
        return self.__flip_percent

    @classmethod
    def __transform_target(
        cls, flipped_indices: dict[int, TargetType], target: TargetType, index: int
    ) -> TargetType:
        if index in flipped_indices:
            return DatasetUtil.replace_target(target, flipped_indices[index])
        return target

    def sample(self, part_index: int) -> DatasetCollection:
        dc = super().sample(part_index=part_index)
        for phase in MachineLearningPhase:
            if phase == MachineLearningPhase.Test:
                continue
            sampler = DatasetSampler(dc.get_dataset_util(phase))
            flip_percent = self.get_flip_percent(part_index=part_index)
            indices = self._dataset_indices[phase][part_index].indices
            assert indices
            assert isinstance(dc, ClassificationDatasetCollection)
            flipped_indices = sampler.randomize_label_by_class(
                percent=flip_percent,
            )

            dc.append_transform(
                transform=functools.partial(self.__transform_target, flipped_indices),
                key=TransformType.Target,
                phases=[phase],
            )
        return dc


class IIDSplitWithSample(IIDSplit):
    def __init__(
        self,
        dataset_collection: ClassificationDatasetCollection,
        part_number: int,
        sample_probs: list[dict[TargetType, float]],
    ) -> None:
        assert len(sample_probs) == part_number
        super().__init__(dataset_collection=dataset_collection, part_number=part_number)
        for phase, part_indices in self._dataset_indices.items():
            for part_index, sample_info in part_indices.items():
                indices = sample_info.indices
                assert indices is not None
                self._samplers[phase].checked_indices = set(indices)
                part_indices[part_index] = SampleInfo(
                    indices=self._samplers[phase].sample_indices(
                        parts=[sample_probs[part_index]],
                    )[0]
                )


class RandomSplitByLabel(SplitBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        parts: list[float] = [1] * self._part_number
        for phase in MachineLearningPhase:
            self.set_split_indices(
                phase=phase,
                index_result=dict(
                    enumerate(
                        self._samplers[phase].random_split_indices(parts, by_label=True)
                    )
                ),
            )


class RandomSplit(SplitBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        parts: list[float] = [1] * self._part_number
        for phase, sample in self._samplers.items():
            self.set_split_indices(
                phase=phase,
                index_result=dict(
                    enumerate(sample.random_split_indices(parts, by_label=False))
                ),
            )


class ProbabilitySampler(SamplerBase):
    def __init__(
        self,
        dataset_collection: DatasetCollection,
        sample_prob: dict[TargetType, float],
    ) -> None:
        super().__init__(dataset_collection=dataset_collection)
        for phase in MachineLearningPhase:
            self._sample_info[phase] = SampleInfo(
                indices=self._samplers[phase].sample_indices(parts=[sample_prob])[0]
            )


global_sampler_factory = Factory()
global_sampler_factory.register("iid", IIDSplit)
global_sampler_factory.register("iid_split_and_flip", IIDSplitWithFlip)
global_sampler_factory.register("iid_split_and_sample", IIDSplitWithSample)
global_sampler_factory.register("random", RandomSplit)
global_sampler_factory.register("prob_sampler", ProbabilitySampler)


def get_dataset_collection_split(
    name: str, dataset_collection: DatasetCollection, **kwargs
) -> SplitBase:
    constructor = global_sampler_factory.get(name.lower())
    if constructor is None:
        raise NotImplementedError(name)
    return constructor(dataset_collection=dataset_collection, **kwargs)


def get_dataset_collection_sampler(
    name: str, dataset_collection: DatasetCollection, **kwargs
) -> SamplerBase:
    constructor = global_sampler_factory.get(name.lower())
    if constructor is None:
        raise NotImplementedError(name)
    return constructor(dataset_collection=dataset_collection, **kwargs)
