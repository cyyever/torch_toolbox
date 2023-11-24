from enum import StrEnum, auto


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
    MODEL_BACKWARD = auto()
    OPTIMIZER_STEP = auto()
    BEFORE_FETCH_BATCH = auto()
    AFTER_FETCH_BATCH = auto()
    BEFORE_BATCH = auto()
    AFTER_FORWARD = auto()
    AFTER_BATCH = auto()
    AFTER_VALIDATION = auto()


class StopExecutingException(Exception):
    pass
