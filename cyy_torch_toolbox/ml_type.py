from enum import IntEnum, auto


class MachineLearningPhase(IntEnum):
    Training = auto()
    Validation = auto()
    Test = auto()


class ModelType(IntEnum):
    Classification = auto()
    Detection = auto()


class DatasetType(IntEnum):
    Vision = auto()
    Text = auto()
    Audio = auto()


class ModelExecutorHookPoint(IntEnum):
    BEFORE_EXECUTE = auto()
    AFTER_EXECUTE = auto()
    BEFORE_EPOCH = auto()
    AFTER_EPOCH = auto()
    OPTIMIZER_STEP = auto()
    AFTER_OPTIMIZER_STEP = auto()
    BEFORE_FETCH_BATCH = auto()
    AFTER_FETCH_BATCH = auto()
    BEFORE_BATCH = auto()
    AFTER_BATCH = auto()
    AFTER_LOAD_MODEL = auto()
    AFTER_VALIDATION = auto()


class StopExecutingException(Exception):
    pass
