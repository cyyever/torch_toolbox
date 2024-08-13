from enum import StrEnum, auto

from .config import ConfigBase


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
]
