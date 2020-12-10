from hyper_parameter import HyperParameter
from trainer import Trainer, ClassificationTrainer, DetectionTrainer
from dataset import get_dataset
from local_types import MachineLearningPhase
from hyper_parameter import get_recommended_hyper_parameter
from model_factory import get_model
from local_types import ModelType


def get_trainer_from_configuration(
    dataset_name: str, model_name: str, hyper_parameter: HyperParameter = None
) -> Trainer:
    if hyper_parameter is None:
        hyper_parameter = get_recommended_hyper_parameter(
            dataset_name, model_name)
        assert hyper_parameter is not None

    training_dataset = get_dataset(dataset_name, MachineLearningPhase.Training)
    validation_dataset = get_dataset(
        dataset_name, MachineLearningPhase.Validation)
    test_dataset = get_dataset(dataset_name, MachineLearningPhase.Test)
    model_with_loss = get_model(model_name, training_dataset)
    trainer: Trainer = None
    if model_with_loss.model_type == ModelType.Classification:
        trainer = ClassificationTrainer(
            model_with_loss, training_dataset, hyper_parameter
        )
    elif model_with_loss.model_type == ModelType.Detection:
        trainer = DetectionTrainer(
            model_with_loss,
            training_dataset,
            hyper_parameter)
    trainer.set_validation_dataset(validation_dataset)
    trainer.set_test_dataset(test_dataset)
    return trainer
