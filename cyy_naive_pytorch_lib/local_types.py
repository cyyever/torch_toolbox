from enum import Enum, auto


class MachineLearningPhase(Enum):
    Training = auto()
    Validation = auto()
    Test = auto()


class ModelType(Enum):
    Classification = auto()
    Detection = auto()
