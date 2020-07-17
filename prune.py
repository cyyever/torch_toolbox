#!/usr/bin/env python3
import os
import datetime
import copy
import torch
import torch.nn.utils.prune

from cyy_naive_lib.log import get_logger

from .visualization import Window, EpochWindow
from .model_util import ModelUtil
from .configuration import get_task_configuration


def lottery_ticket_prune(
    task_name,
    pruning_accuracy,
    pruning_amount,
    model_path=None,
    hyper_parameter=None,
    save_dir=None,
):
    Window.set_env(
        "prune_"
        + task_name
        + "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    )

    trainer = get_task_configuration(task_name, True)
    if model_path is not None:
        get_logger().info("use model %s", model_path)
        trainer.model = torch.load(model_path)

    if hyper_parameter is not None:
        default_hyper_parameter = trainer.get_hyper_parameter()
        if hyper_parameter.epochs is not None:
            default_hyper_parameter.epochs = hyper_parameter.epochs
        if hyper_parameter.batch_size is not None:
            default_hyper_parameter.batch_size = hyper_parameter.batch_size
        if hyper_parameter.learning_rate is not None:
            default_hyper_parameter.learning_rate = hyper_parameter.learning_rate
        if hyper_parameter.weight_decay is not None:
            default_hyper_parameter.weight_decay = hyper_parameter.weight_decay
        trainer.set_hyper_parameter(default_hyper_parameter)
    init_hyper_parameter = copy.deepcopy(trainer.get_hyper_parameter())
    get_logger().info("prune model when test accuracy is %s", pruning_accuracy)
    get_logger().info("prune amount is %s", pruning_amount)

    model_util = ModelUtil(trainer.model)
    init_parameters = model_util.get_original_parameters()
    for k, v in init_parameters.items():
        init_parameters[k] = copy.deepcopy(v)

    if save_dir is None:
        save_dir = os.path.join(
            "models",
            trainer.model.__class__.__name__ +
            "_" +
            task_name,
            "pruned")

    def after_epoch_callback(trainer, epoch, _):
        nonlocal init_parameters
        nonlocal save_dir
        nonlocal init_hyper_parameter
        nonlocal model_util

        parameters = model_util.get_parameter_list().detach().clone().cpu()

        abs_parameters = parameters.abs()
        abs_parameters = abs_parameters[abs_parameters.nonzero()]
        win = EpochWindow("abs parameter statistics")
        win.y_label = "statistics"
        win.plot_scalar(epoch, abs_parameters.mean(), "mean value")
        win.plot_scalar(epoch, abs_parameters.max(), "max value")
        win.plot_scalar(epoch, abs_parameters.min(), "min value")
        win = EpochWindow("abs parameter variance")
        win.y_label = "variance"
        win.plot_scalar(epoch, abs_parameters.var())

        if trainer.validation_accuracy[epoch] < pruning_accuracy:
            return

        pruned_parameters = init_parameters.keys()

        sparsity, _, __ = model_util.get_sparsity()
        get_logger().info("before prune sparsity is %s%%", sparsity)

        torch.nn.utils.prune.global_unstructured(
            pruned_parameters,
            pruning_method=torch.nn.utils.prune.L1Unstructured,
            amount=pruning_amount,
        )

        sparsity, _, __ = model_util.get_sparsity()
        get_logger().info("after prune sparsity is %s%%", sparsity)

        for k, parameter in init_parameters.items():
            layer, name = k
            orig = getattr(layer, name + "_orig")
            orig.data = copy.deepcopy(parameter).data
            # delattr(layer, name + "_orig")
            # layer.register_parameter(
            #     name + "_orig",
            #     torch.nn.Parameter(copy.deepcopy(parameter).to(orig_device)),
            # )

            # mask = getattr(layer, name + "_mask")
            # orig = getattr(layer, name + "_orig")
            # pruned_tensor = mask.to(dtype=orig.dtype) * orig
            # setattr(layer, name, pruned_tensor)
        trainer.set_hyper_parameter(copy.deepcopy(init_hyper_parameter))
        trainer.save(os.path.join(save_dir, str(epoch)))

    trainer.train(plot_parameter_distribution=True,
                  after_epoch_callback=after_epoch_callback)
    trainer.save(save_dir)
