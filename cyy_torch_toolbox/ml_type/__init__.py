from collections.abc import Iterable
from enum import StrEnum, auto

import torch

from .config import ConfigBase
from .factory import Factory

type OptionalTensor = torch.Tensor | None
type TensorDict = dict[str, torch.Tensor]
type OptionalTensorDict = TensorDict | None
type BlockType = list[tuple[str, torch.nn.Module]]
type IndicesType = Iterable[int]
type OptionalIndicesType = IndicesType | None
type ModelGradient = TensorDict
type SampleTensors = dict[int, torch.Tensor]
type SampleGradients = dict[int, ModelGradient]
type ModelParameter = TensorDict


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
    Seq2SeqLM = auto()
    TokenClassification = auto()
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
