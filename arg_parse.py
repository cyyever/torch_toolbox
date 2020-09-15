#!/usr/bin/env python3
import uuid
import os
import copy
import json
import argparse
import random
import torch
import numpy as np

from cyy_naive_lib.log import get_logger

from tools.dataset import (
    dataset_with_indices,
    sample_subset,
    sub_dataset,
    replace_dataset_labels,
    DatasetType,
    get_dataset,
)
from tools.configuration import get_task_configuration, get_task_dataset_name
from tools.hyper_gradient_trainer import HyperGradientTrainer


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--stop_accuracy", type=float, default=None)
    parser.add_argument("--cache_size", type=int, default=1024)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--approx_hyper_gradient_and_momentum_dir", type=str, default=None
    )
    parser.add_argument(
        "--hessian_hyper_gradient_and_momentum_dir", type=str, default=None
    )
    parser.add_argument(
        "--hyper_gradient_sample_percentage",
        type=float,
        default=None)
    parser.add_argument(
        "--training_dataset_percentage",
        type=float,
        default=None)
    parser.add_argument("--use_hessian", action="store_true", default=False)
    parser.add_argument(
        "--use_hessian_and_approximation", action="store_true", default=False
    )
    parser.add_argument("--repeated_num", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--reproducing_seed", type=int, default=None)
    return parser


def get_parsed_args(parser=None):
    if parser is None:
        parser = get_arg_parser()
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join("models", args.task_name)
    return set_save_dir_of_args(args, args.save_dir)


def set_save_dir_of_args(args, save_dir: str):
    args.save_dir = os.path.join(save_dir, str(uuid.uuid4()))
    return args


def create_trainer_from_args(args):
    if args.reproducing_seed is not None:
        get_logger().warning("set reproducing seed")
        assert isinstance(args.reproducing_seed, int)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.reproducing_seed)
        random.seed(args.reproducing_seed)
        np.random.seed(args.reproducing_seed)

    trainer = get_task_configuration(args.task_name, True)
    if args.model_path is not None:
        trainer.load_model(args.model_path)

    hyper_parameter = copy.deepcopy(trainer.get_hyper_parameter())
    if args.epochs is not None:
        hyper_parameter.epochs = args.epochs
    if args.batch_size is not None:
        hyper_parameter.batch_size = args.batch_size
    if args.learning_rate is not None:
        hyper_parameter.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        hyper_parameter.weight_decay = args.weight_decay
    trainer.set_hyper_parameter(hyper_parameter)

    if args.training_dataset_percentage is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        subset_dict = sample_subset(
            trainer.training_dataset, args.training_dataset_percentage
        )
        sample_indices = sum(subset_dict.values(), [])
        trainer.training_dataset = sub_dataset(
            trainer.training_dataset, sample_indices)
        with open(
            os.path.join(args.save_dir, "training_dataset_indices.json"),
            mode="wt",
        ) as f:
            json.dump(sample_indices, f)

    trainer.training_dataset = dataset_with_indices(trainer.training_dataset)

    if args.stop_accuracy is not None:
        trainer.stop_criterion = (
            lambda trainer, epoch, __: trainer.validation_accuracy[epoch]
            >= args.stop_accuracy
        )
    return trainer


def create_hyper_gradient_trainer_from_args(args):
    trainer = create_trainer_from_args(args)

    use_approximation = True
    if args.use_hessian:
        use_approximation = False
    if args.use_hessian_and_approximation:
        args.use_hessian = True
        use_approximation = True

    hyper_gradient_trainer = HyperGradientTrainer(
        trainer,
        args.cache_size,
        args.save_dir,
        hessian_hyper_gradient_and_momentum_dir=args.hessian_hyper_gradient_and_momentum_dir,
        approx_hyper_gradient_and_momentum_dir=args.approx_hyper_gradient_and_momentum_dir,
        use_hessian=args.use_hessian,
        use_approximation=use_approximation,
    )

    if args.hyper_gradient_sample_percentage is not None:
        subset_dict = sample_subset(
            trainer.training_dataset,
            args.hyper_gradient_sample_percentage,
        )
        sample_indices = sum(subset_dict.values(), [])
        os.makedirs(args.save_dir, exist_ok=True)
        with open(
            os.path.join(args.save_dir, "hyper_gradient_indices.json"),
            mode="wt",
        ) as f:
            json.dump(sample_indices, f)
        hyper_gradient_trainer.set_computed_indices(sample_indices)
    return hyper_gradient_trainer


def create_validator_from_args(args):
    validator = get_task_configuration(args.task_name, False)
    if args.model_path is not None:
        validator.load_model(args.model_path)
    return validator


def get_randomized_label_map(args):
    randomized_label_map: dict = dict()
    with open(args.randomized_label_map_path, "r") as f:
        for k, v in json.load(f).items():
            randomized_label_map[int(k)] = int(v)
    return randomized_label_map


def get_training_dataset(args):
    dataset_name = get_task_dataset_name(args.task_name)
    training_dataset = get_dataset(dataset_name, DatasetType.Training)
    if (
        hasattr(args, "training_dataset_indices_path")
        and args.training_dataset_indices_path is not None
    ):
        get_logger().info("use training_dataset_indices_path")
        with open(args.training_dataset_indices_path, "r") as f:
            subset_indices = json.load(f)
            training_dataset = sub_dataset(training_dataset, subset_indices)
    if args.randomized_label_map_path is not None:
        get_logger().info("use randomized_label_map_path")
        training_dataset = replace_dataset_labels(
            training_dataset, get_randomized_label_map(args)
        )
    return training_dataset
