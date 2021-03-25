#!/usr/bin/env python3
import argparse
import json
import os

from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.dataset import DatasetUtil
from cyy_naive_pytorch_lib.default_config import DefaultConfig

from .hydra_callback import HyDRACallback


class HyDRAConfig(DefaultConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_size: int = None
        self.approx_hyper_gradient_and_momentum_dir: str = None
        self.hessian_hyper_gradient_and_momentum_dir: str = None
        self.tracking_percentage: float = None
        self.tracking_indices = None
        self.use_hessian: bool = False
        self.use_approximation: bool = True

    # def load_args(self, parser=None):
    #     if parser is None:
    #         parser = argparse.ArgumentParser()
    #     parser.add_argument(
    #         "--algo.hydra.use_hessian", action="store_true", default=False
    #     )
    #     parser.add_argument(
    #         "--algo.hydra.no_approximation", action="store_true", default=False
    #     )
    #     parser.add_argument("--algo.hydra.", type=float, default=None)
    #     super().load_args(parser=parser)

    def create_trainer(self, **kwargs):
        trainer = super().create_trainer(**kwargs)

        hydra_callback = HyDRACallback(
            self.cache_size,
            self.get_save_dir(),
            hessian_hyper_gradient_and_momentum_dir=self.hessian_hyper_gradient_and_momentum_dir,
            approx_hyper_gradient_and_momentum_dir=self.approx_hyper_gradient_and_momentum_dir,
            use_hessian=self.use_hessian,
            use_approximation=self.use_approximation,
        )
        hydra_callback.append_to_model_executor(trainer)

        if self.tracking_percentage is not None:
            subset_dict = DatasetUtil(trainer.dataset).sample_subset(
                self.tracking_percentage
            )
            self.tracking_indices = sum(subset_dict.values(), [])
        if self.tracking_indices:
            with open(
                os.path.join(self.get_save_dir(), "hyper_gradient_indices.json"),
                mode="wt",
            ) as f:
                json.dump(self.tracking_indices, f)
            get_logger().info("track %s samples", len(self.tracking_indices))
            hydra_callback.set_computed_indices(self.tracking_indices)
        return trainer
