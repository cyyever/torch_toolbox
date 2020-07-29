import uuid
import copy
import os
import shutil
import torch
import torch.nn.utils.prune as prune
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter
from cyy_naive_lib.list_op import split_list_to_chunks
import cyy_pytorch_cpp

from model_util import ModelUtil
from .hessian_vector_product import get_hessian_vector_product_func


class HyperGradientTrainer:
    def __init__(self, trainer, cache_size, save_dir, **kwargs):
        self.trainer = trainer
        self.save_dir = save_dir

        hyper_gradient_matrix_dir = kwargs.get(
            "hyper_gradient_matrix_dir", None)
        if hyper_gradient_matrix_dir is not None:
            self.hyper_gradient_matrix = HyperGradientTrainer.create_gradient_matrix(
                cache_size, trainer.model, storage_dir=hyper_gradient_matrix_dir)
        else:
            self.hyper_gradient_matrix = HyperGradientTrainer.create_gradient_matrix(
                cache_size, trainer.model)
            self.hyper_gradient_matrix.set_storage_dir(os.path.join(
                save_dir, "hyper_gradient_matrix", str(uuid.uuid4()),))
        get_logger().info(
            "use hyper_gradient_matrix_dir:%s",
            self.hyper_gradient_matrix.get_storage_dir(),
        )

        mom_gradient_matrix_dir = kwargs.get("mom_gradient_matrix_dir", None)
        if mom_gradient_matrix_dir is not None:
            self.mom_gradient_matrix = HyperGradientTrainer.create_gradient_matrix(
                cache_size, trainer.model, storage_dir=mom_gradient_matrix_dir)
            get_logger().info(
                "use mom_gradient_matrix_dir :%s", mom_gradient_matrix_dir
            )
        else:
            self.mom_gradient_matrix = HyperGradientTrainer.create_gradient_matrix(
                cache_size, trainer.model)
            self.mom_gradient_matrix.set_storage_dir(os.path.join(
                save_dir, "mom_gradient_matrix", str(uuid.uuid4()),))
        get_logger().info(
            "use mom_gradient_matrix_dir:%s",
            self.mom_gradient_matrix.get_storage_dir(),
        )
        self.batch_gradients = dict()
        self.computed_indices = None
        self.delayed_computations = dict()
        self.use_hessian = kwargs.get("use_hessian", False)
        if self.use_hessian:
            get_logger().info("use hessian to compute hyper-gradients")
        self.hvp_function = None

    def train(self, computed_indices=None):
        get_logger().info("begin train")

        if computed_indices is not None:
            self.computed_indices = set(computed_indices)
        else:
            self.computed_indices = None

        self.delayed_computations = dict()
        for k in self.__get_real_computed_indices():
            self.delayed_computations[str(k)] = []

        self.trainer.train(
            pre_batch_callback=self.__pre_batch_callback,
            per_sample_gradient_callback=(
                self.__per_sample_gradient_callback,
                self.computed_indices,
            ),
            after_batch_callback=self.__after_batch_callback,
            after_epoch_callback=self.__after_epoch_callback,
        )
        get_logger().info("begin do __do_delayed_computation")
        self.__do_delayed_computation()
        get_logger().info("end do __do_delayed_computation")
        self.trainer.save(self.save_dir)
        self.trainer.save(self.hyper_gradient_matrix.get_storage_dir())
        self.hyper_gradient_matrix.flush_all()
        self.hyper_gradient_matrix.release()
        self.mom_gradient_matrix.release()

    def __do_delayed_computation_with_hessian(self):
        for chunk in split_list_to_chunks(
            [str(idx) for idx in sorted(list(self.__get_real_computed_indices()))], 100
        ):
            counter = TimeCounter()
            self.hyper_gradient_matrix.prefetch(chunk)
            self.mom_gradient_matrix.prefetch(chunk)
            hyper_gradients = list()
            hyper_gradient_indices = list()
            hessian_vector_product_dict = dict()
            for index in chunk:
                if index in self.hyper_gradient_matrix:
                    hyper_gradients.append(self.hyper_gradient_matrix[index])
                    hyper_gradient_indices.append(index)
            if hyper_gradients:
                counter2 = TimeCounter()
                hessian_vector_products = self.hvp_function(hyper_gradients)
                get_logger().info(
                    "hvp chunk size %s use time %s ms",
                    len(hyper_gradients),
                    counter2.elapsed_milliseconds(),
                )

                assert len(hyper_gradients) == len(hessian_vector_products)
                for idx, hessian_vector_product in zip(
                    hyper_gradient_indices, hessian_vector_products
                ):
                    hessian_vector_product_dict[idx] = hessian_vector_product

            for index in chunk:
                assert len(self.delayed_computations[index]) == 1
                (
                    momentum,
                    weight_decay,
                    learning_rate,
                    instance_gradient,
                ) = self.delayed_computations[index][0]

                hyper_gradient = None
                if index in self.hyper_gradient_matrix:
                    hyper_gradient = self.hyper_gradient_matrix[index]

                mom_gradient = None
                if index in self.mom_gradient_matrix:
                    mom_gradient = self.mom_gradient_matrix[index]

                if mom_gradient is not None:
                    mom_gradient *= momentum

                if hyper_gradient is not None:
                    res = weight_decay * hyper_gradient
                    res += hessian_vector_product_dict[index]
                    if mom_gradient is not None:
                        mom_gradient += res
                    else:
                        mom_gradient = res

                if instance_gradient is not None:
                    if mom_gradient is not None:
                        mom_gradient += instance_gradient
                    else:
                        mom_gradient = instance_gradient

                if mom_gradient is not None:
                    if hyper_gradient is not None:
                        hyper_gradient -= learning_rate * mom_gradient
                    else:
                        hyper_gradient = -learning_rate * mom_gradient

                if mom_gradient is not None:
                    self.mom_gradient_matrix[index] = mom_gradient
                if hyper_gradient is not None:
                    self.hyper_gradient_matrix[index] = hyper_gradient
                self.delayed_computations[index] = []
            get_logger().info(
                "__do_delayed_computation_with_hessian chunk size %s use time %s ms",
                len(chunk),
                counter.elapsed_milliseconds(),
            )

    def __do_delayed_computation(self, index=None):
        if index is None:
            fast_keys = self.hyper_gradient_matrix.in_memory_keys()
            get_logger().info(
                "begin do __do_delayed_computation from fast keys %s",
                len(fast_keys))
            for k in fast_keys:
                if k in self.delayed_computations and self.delayed_computations[k]:
                    self.__do_delayed_computation(k)
            get_logger().info("end do __do_delayed_computation from fast keys")

            unfinished_keys = []
            for k, v in self.delayed_computations.items():
                if v:
                    unfinished_keys.append(k)

            for chunk in split_list_to_chunks(unfinished_keys, 100):
                self.hyper_gradient_matrix.prefetch(chunk)
                self.mom_gradient_matrix.prefetch(chunk)
                for k in chunk:
                    self.__do_delayed_computation(k)
            return

        mom_gradient = None
        if index in self.mom_gradient_matrix:
            mom_gradient = self.mom_gradient_matrix[index]

        hyper_gradient = None
        if index in self.hyper_gradient_matrix:
            hyper_gradient = self.hyper_gradient_matrix[index]

        for arguments in self.delayed_computations[index]:
            (momentum, weight_decay, learning_rate, instance_gradient,) = arguments
            if mom_gradient is not None:
                mom_gradient *= momentum

            if hyper_gradient is not None:
                res = weight_decay * hyper_gradient
                if mom_gradient is not None:
                    mom_gradient += res
                else:
                    mom_gradient = res

            if instance_gradient is not None:
                if mom_gradient is not None:
                    mom_gradient += instance_gradient
                else:
                    mom_gradient = instance_gradient

            if mom_gradient is not None:
                if hyper_gradient is not None:
                    hyper_gradient -= learning_rate * mom_gradient
                else:
                    hyper_gradient = -learning_rate * mom_gradient

        assert mom_gradient is not None
        assert hyper_gradient is not None
        self.delayed_computations[index] = []
        self.mom_gradient_matrix[index] = mom_gradient
        self.hyper_gradient_matrix[index] = hyper_gradient

    @staticmethod
    def create_gradient_matrix(
        cache_size, model, storage_dir="",
    ):

        mask = None
        gradient_shape = None
        if prune.is_pruned(model):
            model_util = ModelUtil(model)
            get_logger().info(
                "use pruned model, sparsity is %s",
                model_util.get_sparsity()[0])
            parameters = model_util.get_parameter_list()
            gradient_shape = parameters.shape
            mask = model_util.get_pruning_mask_list()
            assert len(mask) == len(parameters)
        m = None
        if mask is not None:
            m = cyy_pytorch_cpp.data_structure.SyncedSparseTensorDict(
                mask, gradient_shape, storage_dir
            )
        else:
            m = cyy_pytorch_cpp.data_structure.SyncedTensorDict(storage_dir)
        m.set_permanent_storage()
        m.set_in_memory_number(cache_size)
        get_logger().info("gradient matrix use cache size %s", cache_size)
        m.set_logging(False)
        return m

    def __pre_batch_callback(self, trainer, batch, batch_index):
        batch_gradient_indices = {i.data.item() for i in batch[2]}

        if self.computed_indices is not None:
            batch_gradient_indices &= self.computed_indices

        self.hyper_gradient_matrix.prefetch(
            [str(i) for i in batch_gradient_indices])
        self.mom_gradient_matrix.prefetch(
            [str(i) for i in batch_gradient_indices])
        self.batch_gradients.clear()

        if self.use_hessian:
            self.hvp_function = get_hessian_vector_product_func(
                trainer.model, batch, trainer.loss_fun
            )

    def __per_sample_gradient_callback(
        self, trainer, instance_index, instance_gradient, **kwargs,
    ):
        assert instance_index in self.__get_real_computed_indices()
        self.batch_gradients[str(instance_index)] = instance_gradient

    def __get_real_computed_indices(self):
        if self.computed_indices is not None:
            return self.computed_indices
        return range(len(self.trainer.training_dataset))

    def __after_batch_callback(
        self,
        trainer,
        epoch,
        batch_index,
        batch_size,
        batch_loss,
        cur_learning_rates,
        **kwargs,
    ):

        optimizer = kwargs["optimizer"]
        if not isinstance(optimizer, torch.optim.SGD):
            raise RuntimeError("not SGD")

        cur_learning_rate = cur_learning_rates[0]

        momentums = [group["momentum"] for group in optimizer.param_groups]
        if len(momentums) != 1:
            raise RuntimeError("unsupported momentums")

        momentum = momentums[0]
        weight_decay = trainer.get_hyper_parameter().weight_decay

        training_set_size = len(trainer.training_dataset)

        for idx in self.__get_real_computed_indices():
            idx = str(idx)
            instance_gradient = None
            if idx in self.batch_gradients:
                instance_gradient = (
                    (self.batch_gradients[idx] *
                     training_set_size /
                     batch_size) .detach() .clone())
            self.delayed_computations[idx].append(
                (momentum, weight_decay, cur_learning_rate, instance_gradient)
            )
        if self.use_hessian:
            self.__do_delayed_computation_with_hessian()
        else:
            for idx in self.__get_real_computed_indices():
                idx = str(idx)
                if idx in self.batch_gradients:
                    self.__do_delayed_computation(idx)

    def __after_epoch_callback(self, trainer, epoch, cur_learning_rates):
        total_epochs = trainer.get_hyper_parameter().epochs
        if epoch == total_epochs:
            return
        if epoch % 10 != 0:
            return
        cur_accurary = trainer.validation_accuracy[epoch]
        validation_accuracy = copy.deepcopy(trainer.validation_accuracy)
        validation_accuracy.pop(epoch)
        max_accuracy = max(validation_accuracy.values())
        if cur_accurary < max_accuracy + 0.01:
            return
        get_logger().info("begin do __do_delayed_computation")
        self.__do_delayed_computation()
        get_logger().info("end do __do_delayed_computation")
        self.hyper_gradient_matrix.flush_all()
        self.mom_gradient_matrix.flush_all()
        self.hyper_gradient_matrix.flush_all(True)
        shutil.copytree(
            self.hyper_gradient_matrix.get_storage_dir(),
            self.hyper_gradient_matrix.get_storage_dir() +
            "_epoch_" +
            str(epoch),
        )
        self.mom_gradient_matrix.flush_all(True)
        shutil.copytree(
            self.mom_gradient_matrix.get_storage_dir(),
            self.mom_gradient_matrix.get_storage_dir() +
            "_epoch_" +
            str(epoch),
        )
        epoch_save_dir = os.path.join(self.save_dir, "epoch_" + str(epoch))
        trainer.save(epoch_save_dir)
