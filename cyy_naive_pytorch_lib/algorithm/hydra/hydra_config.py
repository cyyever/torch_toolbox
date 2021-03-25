#!/usr/bin/env python3
import json
import os

from cyy_naive_lib.log import get_logger

from config import Config
from dataset import DatasetUtil

from .hyper_gradient_callback import HyperGradientCallback


class HydraConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_size: int = None
        self.approx_hyper_gradient_and_momentum_dir: str = None
        self.hessian_hyper_gradient_and_momentum_dir: str = None
        self.hyper_gradient_sample_percentage: float = None
        self.use_hessian: bool = False
        self.use_approximation: bool = True

    def create_trainer(self, **kwargs):
        trainer = super().create_trainer(**kwargs)

        hyper_gradient_callback = HyperGradientCallback(
            self.cache_size,
            self.get_save_dir(),
            hessian_hyper_gradient_and_momentum_dir=self.hessian_hyper_gradient_and_momentum_dir,
            approx_hyper_gradient_and_momentum_dir=self.approx_hyper_gradient_and_momentum_dir,
            use_hessian=self.use_hessian,
            use_approximation=self.use_approximation,
        )
        hyper_gradient_callback.append_to_model_executor(trainer)

        if self.hyper_gradient_sample_percentage is not None:
            subset_dict = DatasetUtil(trainer.dataset).sample_subset(
                self.hyper_gradient_sample_percentage
            )
            sample_indices = sum(subset_dict.values(), [])
            os.makedirs(self.get_save_dir(), exist_ok=True)
            with open(
                os.path.join(self.get_save_dir(), "hyper_gradient_indices.json"),
                mode="wt",
            ) as f:
                json.dump(sample_indices, f)
            get_logger().info("track %s samples", len(sample_indices))
            hyper_gradient_callback.set_computed_indices(sample_indices)
        return trainer
