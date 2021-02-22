from dataset_collection import DatasetCollection
from hyper_parameter import HyperParameter, get_recommended_hyper_parameter
from model_factory import get_model
from trainer import Trainer


def get_trainer_from_configuration(
    dataset_name: str,
    model_name: str,
    hyper_parameter: HyperParameter = None,
    trainer_type=Trainer,
) -> Trainer:
    if hyper_parameter is None:
        hyper_parameter = get_recommended_hyper_parameter(dataset_name, model_name)
        assert hyper_parameter is not None

    dc = DatasetCollection.get_by_name(dataset_name)
    model_with_loss = get_model(model_name, dc)
    trainer = trainer_type(model_with_loss, dc, hyper_parameter)
    return trainer
