from enum import IntEnum, auto


class MachineLearningPhase(IntEnum):
    Training = auto()
    Validation = auto()
    Test = auto()


class ModelType(IntEnum):
    Classification = auto()
    Detection = auto()
