from enum import Enum, IntEnum, auto


class MachineLearningPhase(IntEnum):
    Training = auto()
    Validation = auto()
    Test = auto()


class ModelType(Enum):
    Classification = auto()
    Detection = auto()
