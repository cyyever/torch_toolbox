#!/usr/bin/env python3

import argparse

from cyy_naive_pytorch_lib.dataset import DatasetUtil
from cyy_naive_pytorch_lib.default_config import DefaultConfig

from .hydra_hook import HyDRAHook


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

    def load_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        parser.add_argument("--tracking_percentage", type=float, default=None)
        super().load_args(parser=parser)

    def create_trainer(self, return_hydra_hook=False, **kwargs):
        trainer = super().create_trainer(**kwargs)

        hydra_hook = HyDRAHook(
            self.cache_size,
            self.get_save_dir(),
            hessian_hyper_gradient_and_momentum_dir=self.hessian_hyper_gradient_and_momentum_dir,
            approx_hyper_gradient_and_momentum_dir=self.approx_hyper_gradient_and_momentum_dir,
            use_hessian=self.use_hessian,
            use_approximation=self.use_approximation,
        )
        hydra_hook.append_to_model_executor(trainer)
        hydra_hook.set_stripable(trainer)

        if self.tracking_percentage is not None:
            subset_dict = DatasetUtil(trainer.dataset).iid_sample(
                self.tracking_percentage
            )
            self.tracking_indices = sum(subset_dict.values(), [])
        if self.tracking_indices:
            hydra_hook.set_computed_indices(self.tracking_indices)
        if not return_hydra_hook:
            return trainer
        return trainer, hydra_hook
