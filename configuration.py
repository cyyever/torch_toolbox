from hyper_parameter import HyperParameter
from trainer import Trainer
from inference import Inferencer
from dataset import get_dataset, DatasetType
from hyper_parameter import get_recommended_hyper_parameter
from model_factory import get_model


# def get_task_dataset_name(name):
#     if name.startswith("CIFAR10_"):
#         return "CIFAR10"
#     if name.startswith("MNIST_"):
#         return "MNIST"
#     return name


def get_trainer_from_configuration(
    dataset_name: str, model_name: str, hyper_parameter: HyperParameter = None
):
    if hyper_parameter is None:
        hyper_parameter = get_recommended_hyper_parameter(
            dataset_name, model_name)

    training_dataset = get_dataset(dataset_name, DatasetType.Test)
    validation_dataset = get_dataset(dataset_name, DatasetType.Validation)
    test_dataset = get_dataset(dataset_name, DatasetType.Test)
    trainer = Trainer(get_model(model_name), training_dataset, hyper_parameter)
    trainer.set_validation_dataset(validation_dataset)
    trainer.set_test_dataset(test_dataset)
    return trainer


def get_inferencer_from_configuration(dataset_name: str, model_name: str):
    test_dataset = get_dataset(dataset_name, DatasetType.Test)
    return Inferencer(get_model(model_name), test_dataset)


# def get_task_configuration(task_name: str, for_training: bool):
#     model = None
#     hyper_parameter = None
#     momentum = 0.9
#     if task_name == "MNIST":
#         model = LeNet5()
#         if for_training:
#             hyper_parameter = HyperParameter(
#                 epochs=50, batch_size=64, learning_rate=0.01, weight_decay=1
#             )
#             hyper_parameter.set_lr_scheduler_factory(
#                 lambda optimizer, _, __: optim.lr_scheduler.MultiStepLR(
#                     optimizer, milestones=[5, 55, 555]
#                 )
#             )

#             # hyper_parameter.set_lr_scheduler_factory(
#             #     lambda optimizer, hyper_parameter: optim.lr_scheduler.CosineAnnealingLR(
#             #         optimizer, T_max=hyper_parameter.epochs))

#     elif task_name == "FashionMNIST":
#         model = LeNet5()
#         if for_training:
#             hyper_parameter = HyperParameter(
#                 epochs=50, batch_size=64, learning_rate=0.01, weight_decay=1
#             )
#             hyper_parameter.set_lr_scheduler_factory(
#                 lambda optimizer, _, __: optim.lr_scheduler.ReduceLROnPlateau(
#                     optimizer, verbose=True, factor=0.5, patience=2
#                 )
#             )
#     elif task_name == "CIFAR10":
#         model = densenet_CIFAR10()
#         if for_training:
#             hyper_parameter = HyperParameter(
#                 epochs=350, batch_size=128, learning_rate=0.1, weight_decay=1
#             )

#             hyper_parameter.set_lr_scheduler_factory(
#                 lambda optimizer, _, __: optim.lr_scheduler.ReduceLROnPlateau(
#                     optimizer, verbose=True, factor=0.1
#                 )
#             )
#         raise NotImplementedError(task_name)

#     dataset_name = get_task_dataset_name(task_name)
#     get_logger().info("get dataset %s for task %s", dataset_name, task_name)

#     test_dataset = get_dataset(dataset_name, DatasetType.Test)
#     if for_training:
#         if hyper_parameter.optimizer_factory is None:
#             hyper_parameter.set_optimizer_factory(
#                 lambda params, learning_rate, weight_decay, dataset: optim.SGD(
#                     params,
#                     momentum=momentum,
#                     lr=learning_rate,
#                     weight_decay=(weight_decay / len(dataset)),
#                 )
#             )
#         training_dataset = get_dataset(dataset_name, DatasetType.Training)
#         trainer = Trainer(
#             ModelWithLoss(model),
#             training_dataset,
#             hyper_parameter)
#         trainer.validation_dataset = get_dataset(
#             dataset_name, DatasetType.Validation)
#         trainer.test_dataset = test_dataset
#         return trainer
#     return Validator(ModelWithLoss(model), test_dataset)
