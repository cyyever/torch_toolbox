#!/usr/bin/env python3
import os
import copy
import json
import argparse


from tools.dataset import dataset_with_indices, sample_subset, sub_dataset
from tools.configuration import get_task_configuration
from tools.hyper_gradient_trainer import HyperGradientTrainer


def get_parsed_args():
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
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join("models", args.task_name)
    return args


def create_trainer_from_args(args):
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
        subset_dict = sample_subset(
            trainer.training_dataset, args.training_dataset_percentage
        )
        sample_indices = sum(subset_dict.values(), [])
        trainer.training_dataset = sub_dataset(
            trainer.training_dataset, sample_indices)
        os.makedirs(args.save_dir, exist_ok=True)
        with open(
            os.path.join(args.save_dir, "training_dataset_indices.json"), mode="wt",
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
            trainer.training_dataset, args.hyper_gradient_sample_percentage,
        )
        sample_indices = sum(subset_dict.values(), [])
        os.makedirs(args.save_dir, exist_ok=True)
        with open(
            os.path.join(args.save_dir, "hyper_gradient_indices.json"), mode="wt",
        ) as f:
            json.dump(sample_indices, f)
        hyper_gradient_trainer.set_computed_indices(sample_indices)
    return hyper_gradient_trainer
