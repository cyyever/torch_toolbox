from inverse_hessian_vector_product import stochastic_inverse_hessian_vector_product
from dataset import sub_dataset
from validator import Validator


def compute_classic_influence_function(trainer, sample_index):
    sample_dataset = sub_dataset(trainer.training_dataset, [sample_index])
    validator = Validator(trainer.model, trainer.loss_fun, sample_dataset)
    sample_gradient = validator.get_gradient()
    return stochastic_inverse_hessian_vector_product(
        trainer.model,
        trainer.training_dataset,
        trainer.loss_fun,
        -sample_gradient,
        100,
        trainer.get_hyper_parameter().batch_size,
    )
