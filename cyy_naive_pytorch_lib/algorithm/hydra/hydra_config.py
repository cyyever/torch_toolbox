#!/usr/bin/env python3

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
            subset_dict = DatasetUtil(trainer.dataset).iid_sample(
                self.tracking_percentage
            )
            self.tracking_indices = sum(subset_dict.values(), [])
        if self.tracking_indices:
            hydra_callback.set_computed_indices(self.tracking_indices)
        return trainer
