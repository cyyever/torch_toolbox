from inverse_hessian_vector_product import (
    stochastic_inverse_hessian_vector_product,
    # conjugate_gradient_inverse_hessian_vector_product,
)
from dataset import sub_dataset
from validator import Validator


def compute_classic_influence_function(
        trainer,
        sample_index,
        dampling_term=0,
        scale=1):
    sample_dataset = sub_dataset(trainer.training_dataset, [sample_index])
    validator = Validator(trainer.model, trainer.loss_fun, sample_dataset)
    sample_gradient = validator.get_gradient()

    return stochastic_inverse_hessian_vector_product(
        trainer.model,
        trainer.training_dataset,
        trainer.loss_fun,
        -sample_gradient,
        repeated_num=5,
        max_iteration=10000,
        batch_size=trainer.get_hyper_parameter().batch_size,
        dampling_term=dampling_term,
        scale=scale,
    )
