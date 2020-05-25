#!/usr/bin/env python3
import os
import datetime
import copy
import torch

from .visualization import Window
from .util import (
    model_parameters_to_vector,
    get_pruned_parameters,
    get_model_sparsity,
)
from .configuration import get_task_configuration
from .log import get_logger


def lottery_ticket_prune(
    task_name,
    model_path,
    pruning_accuracy,
    pruning_amount,
    hyper_parameter=None,
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
    get_logger().info("prune model when test accuracy is %s", pruning_accuracy)
    get_logger().info("prune amount is %s", pruning_amount)

    init_parameters = get_pruned_parameters(trainer.model)
    for k, v in init_parameters.items():
        init_parameters[k] = copy.deepcopy(v)

    parameters_size = sum([len(v[0].view(-1))
                           for v in init_parameters.values()])
    assert parameters_size == len(model_parameters_to_vector(trainer.model))
    save_dir = os.path.join(
        "models", trainer.model.__class__.__name__ + "_" + task_name, "pruned"
    )

    def after_epoch_callback(trainer, epoch, _):
        nonlocal init_parameters
        nonlocal save_dir

        parameters = model_parameters_to_vector(
            trainer.model).detach().clone().cpu()

        abs_parameters = parameters.abs()
        abs_parameters = abs_parameters[abs_parameters.nonzero()]
        win = Window.get("abs parameter statistics")
        win.y_label = "statistics"
        win.plot_scalar_by_epoch(epoch, abs_parameters.mean(), "mean value")
        win.plot_scalar_by_epoch(epoch, abs_parameters.max(), "max value")
        win.plot_scalar_by_epoch(epoch, abs_parameters.min(), "min value")
        win = Window.get("abs parameter variance")
        win.y_label = "variance"
        win.plot_scalar_by_epoch(epoch, abs_parameters.var())

        # win = Window.get("parameter and layer")
        # win.x_label = "parameter"
        # win.y_label = "layer"
        # data = None
        # pruned_parameters = get_pruned_parameters(trainer.model)
        # for v in pruned_parameters.values():
        #     parameter, mask, layer_index = v
        #     masked_parameter = parameter.view(-1)
        #     if mask is not None:
        #         masked_parameter = parameter.masked_select(mask).view(-1)

        #     res = torch.stack(
        #         (masked_parameter,
        #          torch.full_like(
        #              masked_parameter,
        #              layer_index)))
        #     if data is None:
        #         data = res
        #     else:
        #         data = torch.cat((data, res), dim=1)
        # win.plot_scatter(data.t())

        if trainer.validation_accuracy[epoch] < pruning_accuracy:
            return

        pruned_parameters = init_parameters.keys()

        sparsity, _, __ = get_model_sparsity(trainer.model)
        get_logger().info("before prune sparsity is %s%%", sparsity)

        torch.nn.utils.prune.global_unstructured(
            pruned_parameters,
            pruning_method=torch.nn.utils.prune.L1Unstructured,
            amount=pruning_amount,
        )

        sparsity, _, __ = get_model_sparsity(trainer.model)
        get_logger().info("after prune sparsity is %s%%", sparsity)

        for k, v in init_parameters.items():
            layer, name = k
            parameter = v[0]
            orig_device = getattr(layer, name + "_orig").device
            delattr(layer, name + "_orig")
            layer.register_parameter(
                name + "_orig",
                torch.nn.Parameter(copy.deepcopy(parameter).to(orig_device)),
            )

            mask = getattr(layer, name + "_mask")
            orig = getattr(layer, name + "_orig")
            pruned_tensor = mask.to(dtype=orig.dtype) * orig
            setattr(layer, name, pruned_tensor)
        trainer.set_hyper_parameter(copy.deepcopy(hyper_parameter))
        trainer.save(os.path.join(save_dir, str(epoch)))

    trainer.train(plot_parameter_distribution=True,
                  after_epoch_callback=after_epoch_callback)
    trainer.save(save_dir)
