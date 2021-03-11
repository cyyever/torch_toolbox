from dataset_collection import DatasetCollection
from hyper_parameter import HyperParameter, get_recommended_hyper_parameter
from inference import Inferencer
from ml_type import MachineLearningPhase
from model_factory import get_model
from trainer import Trainer


def get_trainer_from_configuration(
    dataset_name: str,
    model_name: str,
    hyper_parameter: HyperParameter = None,
) -> Trainer:
    if hyper_parameter is None:
        hyper_parameter = get_recommended_hyper_parameter(dataset_name, model_name)
        assert hyper_parameter is not None

    dc = DatasetCollection.get_by_name(dataset_name)
    model_with_loss = get_model(model_name, dc)
    return Trainer(model_with_loss, dc, hyper_parameter)


def get_inferencer_from_configuration(
    dataset_name: str,
    model_name: str,
    phase: MachineLearningPhase,
    hyper_parameter: HyperParameter = None,
) -> Trainer:
    if hyper_parameter is None:
        hyper_parameter = get_recommended_hyper_parameter(dataset_name, model_name)
        assert hyper_parameter is not None

    dc = DatasetCollection.get_by_name(dataset_name)
    model_with_loss = get_model(model_name, dc)
    return Inferencer(model_with_loss, dc, phase, hyper_parameter)
