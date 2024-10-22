from collections.abc import Iterable
from enum import StrEnum, auto
from typing import TypeAlias

import torch

from .config import ConfigBase
from .factory import Factory

OptionalTensor: TypeAlias = torch.Tensor | None
TensorDict: TypeAlias = dict[str, torch.Tensor]
OptionalTensorDict: TypeAlias = TensorDict | None
BlockType: TypeAlias = list[tuple[str, torch.nn.Module]]
IndicesType: TypeAlias = Iterable[int]
OptionalIndicesType: TypeAlias = IndicesType | None
ModelGradient: TypeAlias = TensorDict
SampleTensors: TypeAlias = dict[int, torch.Tensor]
SampleGradients: TypeAlias = dict[int, ModelGradient]
ModelParameter: TypeAlias = TensorDict


class MachineLearningPhase(StrEnum):
    Training = auto()
    Validation = auto()
    Test = auto()


class EvaluationMode(StrEnum):
    Training = auto()
    Test = auto()
    TestWithGrad = auto()


class ModelType(StrEnum):
    Classification = auto()
    Detection = auto()
    TextGeneration = auto()


class DatasetType(StrEnum):
    Vision = auto()
    Text = auto()
    Graph = auto()
    Audio = auto()
    CodeText = auto()
    Unknown = auto()


class TransformType(StrEnum):
    ExtractData = auto()
    InputText = auto()
    Input = auto()
    RandomInput = auto()
    InputBatch = auto()
    Target = auto()
    TargetBatch = auto()


class ExecutorHookPoint(StrEnum):
    BEFORE_EXECUTE = auto()
    AFTER_EXECUTE = auto()
    BEFORE_EPOCH = auto()
    AFTER_EPOCH = auto()
    MODEL_FORWARD = auto()
    BEFORE_FETCH_BATCH = auto()
    AFTER_FETCH_BATCH = auto()
    BEFORE_BATCH = auto()
    AFTER_BATCH = auto()
    AFTER_VALIDATION = auto()


class StopExecutingException(Exception):
    pass


class IterationUnit(StrEnum):
    Batch = auto()
    Epoch = auto()
    Round = auto()


__all__ = [
    "MachineLearningPhase",
    "EvaluationMode",
    "ModelType",
    "DatasetType",
    "TransformType",
    "ExecutorHookPoint",
    "StopExecutingException",
    "IterationUnit",
    "ConfigBase",
    "BlockType",
    "IndicesType",
    "ModelGradient",
    "ModelParameter",
    "OptionalIndicesType",
    "Factory",
    "OptionalTensor",
    "OptionalTensorDict",
    "SampleGradients",
    "SampleTensors",
    "TensorDict",
]
