#!/usr/bin/env python3
import json
import os

from cyy_naive_lib.log import get_logger

from arg_parse import create_trainer_from_args, get_arg_parser
from dataset import DatasetUtil

from .hyper_gradient_trainer import HyperGradientTrainer


def add_arguments_to_parser(parser=None):
    if parser is None:
        parser = get_arg_parser()

    parser.add_argument("--cache_size", type=int, default=None)
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
    parser.add_argument("--use_hessian", action="store_true", default=False)
    parser.add_argument(
        "--use_approximation",
        action="store_true",
        default=True)
    return parser


def create_hyper_gradient_trainer_from_args(args):
    trainer = create_trainer_from_args(args)

    hyper_gradient_trainer = HyperGradientTrainer(
        trainer,
        args.cache_size,
        args.save_dir,
        hessian_hyper_gradient_and_momentum_dir=args.hessian_hyper_gradient_and_momentum_dir,
        approx_hyper_gradient_and_momentum_dir=args.approx_hyper_gradient_and_momentum_dir,
        use_hessian=args.use_hessian,
        use_approximation=args.use_approximation,
    )

    if args.hyper_gradient_sample_percentage is not None:
        subset_dict = DatasetUtil(trainer.training_dataset).sample_subset(
            args.hyper_gradient_sample_percentage
        )
        sample_indices = sum(subset_dict.values(), [])
        os.makedirs(args.save_dir, exist_ok=True)
        with open(
            os.path.join(args.save_dir, "hyper_gradient_indices.json"),
            mode="wt",
        ) as f:
            json.dump(sample_indices, f)
        get_logger().info("track %s samples", len(sample_indices))
        hyper_gradient_trainer.set_computed_indices(sample_indices)
    return hyper_gradient_trainer
