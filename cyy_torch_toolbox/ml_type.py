from enum import IntEnum, auto
from typing import Type

try:
    from enum import StrEnum

    _StrEnum: Type = StrEnum
except BaseException:
    _StrEnum = IntEnum


class MachineLearningPhase(_StrEnum):
    Training = auto()
    Validation = auto()
    Test = auto()


class ModelType(_StrEnum):
    Classification = auto()
    Detection = auto()
    TextGeneration = auto()


class DatasetType(_StrEnum):
    Vision = auto()
    Text = auto()
    Graph = auto()
    Audio = auto()
    Unknown = auto()


class TransformType(_StrEnum):
    ExtractData = auto()
    InputText = auto()
    Input = auto()
    RandomInput = auto()
    InputBatch = auto()
    Target = auto()
    TargetBatch = auto()


class ExecutorHookPoint(_StrEnum):
    BEFORE_EXECUTE = auto()
    AFTER_EXECUTE = auto()
    BEFORE_EPOCH = auto()
    AFTER_EPOCH = auto()
    MODEL_FORWARD = auto()
    MODEL_BACKWARD = auto()
    CANCEL_FORWARD = auto()
    OPTIMIZER_STEP = auto()
    BEFORE_FETCH_BATCH = auto()
    AFTER_FETCH_BATCH = auto()
    BEFORE_BATCH = auto()
    AFTER_FORWARD = auto()
    AFTER_BATCH = auto()
    AFTER_VALIDATION = auto()


class StopExecutingException(Exception):
    pass
