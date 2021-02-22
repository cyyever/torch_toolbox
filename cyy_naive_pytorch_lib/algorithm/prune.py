#!/usr/bin/env python3
import os
import datetime
import copy
import torch
import torch.nn.utils.prune

from cyy_naive_lib.log import get_logger

from visualization import EpochWindow
from model_util import ModelUtil


def lottery_ticket_prune(
    trainer,
    pruning_accuracy,
    pruning_amount,
    model_path=None,
    save_dir=None,
):
    visualization_env = "prune_" + "{date:%Y-%m-%d_%H:%M:%S}".format(
        date=datetime.datetime.now()
    )

    if model_path is not None:
        get_logger().info("use model %s", model_path)
        trainer.model = torch.load(model_path)
        model_util = ModelUtil(trainer.model)
        sparsity, _, __ = model_util.get_sparsity()
        get_logger().info("loaded model sparsity is %s%%", sparsity)

    init_hyper_parameter = copy.deepcopy(trainer.hyper_parameter)
    get_logger().info("prune model when test accuracy is %s", pruning_accuracy)
    get_logger().info("prune amount is %s", pruning_amount)

    init_parameters = model_util.get_original_parameters_for_pruning()
    for k, v in init_parameters.items():
        init_parameters[k] = copy.deepcopy(v)

    if save_dir is None:
        save_dir = os.path.join(
            "models", trainer.model.__class__.__name__ + "_" + "pruned"
        )

    def after_epoch_callback(trainer, epoch):
        nonlocal init_parameters
        nonlocal save_dir
        nonlocal init_hyper_parameter
        nonlocal model_util

        parameters = model_util.get_parameter_list().detach().clone().cpu()

        abs_parameters = parameters.abs()
        abs_parameters = abs_parameters[abs_parameters.nonzero()]
        win = EpochWindow("abs parameter statistics", env=visualization_env)
        win.y_label = "statistics"
        win.plot_scalar(epoch, abs_parameters.mean(), "mean value")
        win.plot_scalar(epoch, abs_parameters.max(), "max value")
        win.plot_scalar(epoch, abs_parameters.min(), "min value")
        win = EpochWindow("abs parameter variance", env=visualization_env)
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
            if not hasattr(layer, name + "_orig"):
                orig = getattr(layer, name)
                orig.data = copy.deepcopy(parameter).to(orig.device).data
                continue
            orig = getattr(layer, name + "_orig")
            orig.data = copy.deepcopy(parameter).to(orig.device).data
            mask = getattr(layer, name + "_mask")
            pruned_tensor = mask.to(dtype=orig.dtype) * orig
            delattr(layer, name)
            setattr(layer, name, pruned_tensor)

        trainer.set_hyper_parameter(copy.deepcopy(init_hyper_parameter))
        trainer.save_model(os.path.join(save_dir, str(epoch)))

    trainer.train(plot_parameter_distribution=True,
                  after_epoch_callbacks=[after_epoch_callback])
    trainer.save_model(save_dir)
