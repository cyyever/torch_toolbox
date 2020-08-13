import copy
import typing
from inverse_hessian_vector_product import (
    stochastic_inverse_hessian_vector_product,
    # conjugate_gradient_inverse_hessian_vector_product,
)
from dataset import sub_dataset


def compute_classic_influence_function(
    trainer,
    validator,
    sample_indices: typing.Sequence,
    batch_size=None,
    dampling_term=0,
    scale=1,
):
    test_gradient = validator.get_gradient()
    training_dataset_size = len(trainer.training_dataset)
    if batch_size is None:
        batch_size = trainer.get_hyper_parameter().batch_size
    product = (
        stochastic_inverse_hessian_vector_product(
            trainer.model,
            trainer.training_dataset,
            trainer.loss_fun,
            test_gradient,
            repeated_num=5,
            max_iteration=10000,
            batch_size=batch_size,
            dampling_term=dampling_term,
            scale=scale,
        )
        / training_dataset_size
    )
    contributions = dict()
    for sample_index in sample_indices:
        sample_dataset = sub_dataset(trainer.training_dataset, [sample_index])
        sample_validator = copy.deepcopy(validator)
        sample_validator.set_dataset(sample_dataset)
        sample_gradient = sample_validator.get_gradient()
        contributions[sample_index] = (product @ sample_gradient).data.item()
    return contributions
