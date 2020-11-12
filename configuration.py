from hyper_parameter import HyperParameter
from trainer import Trainer
from inference import Inferencer
from dataset import get_dataset, DatasetType
from hyper_parameter import get_recommended_hyper_parameter
from model_factory import get_model


def get_trainer_from_configuration(
    dataset_name: str, model_name: str, hyper_parameter: HyperParameter = None
):
    if hyper_parameter is None:
        hyper_parameter = get_recommended_hyper_parameter(
            dataset_name, model_name)

    training_dataset = get_dataset(dataset_name, DatasetType.Test)
    validation_dataset = get_dataset(dataset_name, DatasetType.Validation)
    test_dataset = get_dataset(dataset_name, DatasetType.Test)
    trainer = Trainer(
        get_model(
            model_name,
            training_dataset),
        training_dataset,
        hyper_parameter)
    trainer.set_validation_dataset(validation_dataset)
    trainer.set_test_dataset(test_dataset)
    return trainer


def get_inferencer_from_configuration(dataset_name: str, model_name: str):
    test_dataset = get_dataset(dataset_name, DatasetType.Test)
    return Inferencer(get_model(model_name, test_dataset), test_dataset)
