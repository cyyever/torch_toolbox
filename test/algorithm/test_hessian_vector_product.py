import torch
from cyy_naive_lib.time_counter import TimeCounter

from configuration import get_trainer_from_configuration
from algorithm.hessian_vector_product import get_hessian_vector_product_func
from model_util import ModelUtil

# from cyy_naive_lib.profiling import Profile


def test_hessian_vector_product():
    trainer = get_trainer_from_configuration("MNIST", "LeNet5")
    training_data_loader = torch.utils.data.DataLoader(
        trainer.training_dataset,
        batch_size=16,
        shuffle=True,
    )
    parameter_vector = ModelUtil(trainer.model).get_parameter_list()
    v = torch.ones(parameter_vector.shape)
    for batch in training_data_loader:
        hvp_function = get_hessian_vector_product_func(
            trainer.model_with_loss, batch)
        a = hvp_function([v, 2 * v, 3 * v])
        assert len(a) == 3
        assert torch.linalg.norm(a[1] - 2 * a[0], ord=2).data.item() < 0.0005
        assert torch.linalg.norm(a[2] - 3 * a[0], ord=2).data.item() < 0.0005

        with TimeCounter() as c:
            a = hvp_function(v)
            print("one use time ", c.elapsed_milliseconds())
            print(a)
            c.reset_start_time()
            a = hvp_function([v, 2 * v])
            print("two use time ", c.elapsed_milliseconds())
            print(a)
            c.reset_start_time()
            a = hvp_function([v, 2 * v, 3 * v])
            print("3 use time ", c.elapsed_milliseconds())
            print(a)
            c.reset_start_time()
            a = hvp_function([v] * 4)
            print("4 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 4)
            print("4 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 64)
            print("64 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 64)
            print("64 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 64)
            print("64 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 100)
            print("100 use time ", c.elapsed_milliseconds())
            c.reset_start_time()
            a = hvp_function([v] * 100)
            print("100 use time ", c.elapsed_milliseconds())
            # with Profile():
            #     c.reset_start_time()
            #     a = hvp_function([v] * 100)
            #     print("100 use time ", c.elapsed_milliseconds())
        break
