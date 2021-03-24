import os
import shutil
import uuid

import torch
import torch.nn.utils.prune as prune
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter

from algorithm.hessian_vector_product import get_hessian_vector_product_func
from algorithm.sample_gradient_callback import SampleGradientCallback
from data_structure.synced_tensor_dict import SyncedTensorDict
from dataset import dataset_with_indices
from ml_type import MachineLearningPhase
from model_util import ModelUtil
from tensor import get_batch_size


class HyperGradientCallback(SampleGradientCallback):
    def __init__(self, cache_size, save_dir, **kwargs):
        super().__init__()
        self.cache_size = cache_size
        self.save_dir = save_dir

        self.computed_indices = None
        self.hessian_computation_arguments = None
        self.delayed_approximation_computations = None

        self.use_hessian = kwargs.get("use_hessian", False)
        self.hessian_hyper_gradient_and_momentum_dir = kwargs.get(
            "hessian_hyper_gradient_and_momentum_dir", None
        )
        self.hvp_function = None
        self.hessian_hyper_gradient_mom_dict = None
        self.use_approximation = kwargs.get("use_approximation", None)

        if self.use_approximation is None:
            self.use_approximation = not self.use_hessian

        self.approx_hyper_gradient_and_momentum_dir = kwargs.get(
            "approx_hyper_gradient_and_momentum_dir", None
        )
        self.approx_hyper_gradient_mom_dict = None

    def _before_execute(self, *args, **kwargs):
        trainer = kwargs["model_executor"]
        if not self.computed_indices:
            self.computed_indices = set(range(len(trainer.dataset)))
        trainer.dataset_collection.transform_dataset(
            MachineLearningPhase.Training, dataset_with_indices
        )
        if self.use_hessian:
            get_logger().info("use hessian to compute hyper-gradients")
            self.hessian_hyper_gradient_mom_dict = (
                HyperGradientCallback.create_hypergradient_dict(
                    self.cache_size,
                    trainer.model,
                    storage_dir=self.hessian_hyper_gradient_and_momentum_dir,
                )
            )
            if not self.hessian_hyper_gradient_and_momentum_dir:
                self.hessian_hyper_gradient_mom_dict.tensor_dict.set_storage_dir(
                    os.path.join(
                        self.save_dir,
                        "hessian_hyper_gradient_and_momentum_dir",
                        str(uuid.uuid4()),
                    )
                )
            else:
                model_file = os.path.join(
                    self.hessian_hyper_gradient_and_momentum_dir, "model", "model.pt"
                )
                if os.path.isfile(model_file):
                    trainer.load_model(model_file)

            get_logger().info(
                "use hessian_hyper_gradient_mom_dir:%s",
                self.hessian_hyper_gradient_mom_dict.tensor_dict.get_storage_dir(),
            )
        if self.use_approximation:
            self.approx_hyper_gradient_mom_dict = (
                HyperGradientCallback.create_hypergradient_dict(
                    self.cache_size,
                    trainer.model,
                    storage_dir=self.approx_hyper_gradient_and_momentum_dir,
                )
            )
            if not self.approx_hyper_gradient_and_momentum_dir:
                self.approx_hyper_gradient_mom_dict.set_storage_dir(
                    os.path.join(
                        self.save_dir,
                        "approx_hyper_gradient_and_momentum_dir",
                        str(uuid.uuid4()),
                    )
                )
            else:
                model_file = os.path.join(
                    self.approx_hyper_gradient_and_momentum_dir, "model", "model.pt"
                )
                if os.path.isfile(model_file):
                    trainer.load_model(model_file)
            get_logger().info(
                "use hyper_gradient_mom_dir:%s",
                self.approx_hyper_gradient_mom_dict.get_storage_dir(),
            )
            self.delayed_approximation_computations = dict()
            for k in self.computed_indices:
                self.delayed_approximation_computations[k] = []

    def set_computed_indices(self, computed_indices):
        get_logger().info("only compute %s indices", len(computed_indices))
        self.computed_indices = set(computed_indices)
        super().set_computed_indices(computed_indices)

    def _after_execute(self, *args, **kwargs):
        get_logger().info("end hyper-gradient tracking")
        trainer = kwargs["model_executor"]
        trainer.save_model(self.save_dir)
        if self.use_approximation:
            self.__save_hyper_gradients(
                trainer,
                os.path.join(
                    self.save_dir, "approximation_hyper_gradient_dir", str(uuid.uuid4())
                ),
                use_approximation=True,
            )
            self.approx_hyper_gradient_mom_dict.release()
            shutil.rmtree(self.approx_hyper_gradient_mom_dict.get_storage_dir())
            self.approx_hyper_gradient_mom_dict = None
        if self.use_hessian:
            self.__save_hyper_gradients(
                trainer,
                os.path.join(
                    self.save_dir, "hessian_hyper_gradient_dir", str(uuid.uuid4())
                ),
                use_approximation=False,
            )
            self.hessian_hyper_gradient_mom_dict.release()
            shutil.rmtree(self.hessian_hyper_gradient_mom_dict.get_storage_dir())
            self.hessian_hyper_gradient_mom_dict = None

    def __do_computation_with_hessian(self):
        for chunk in split_list_to_chunks(
            [str(idx) for idx in sorted(list(self.computed_indices))],
            self.cache_size // 2,
        ):
            counter = TimeCounter()
            self.hessian_hyper_gradient_mom_dict.prefetch(chunk)
            hyper_gradients = list()
            hyper_gradient_indices = list()
            hessian_vector_product_dict = dict()
            for index in chunk:
                if index in self.hessian_hyper_gradient_mom_dict:
                    hyper_gradients.append(
                        self.get_hyper_gradient(index, use_approximation=False)
                    )
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
                (
                    momentum,
                    weight_decay,
                    learning_rate,
                    instance_gradient,
                ) = self.hessian_computation_arguments[index]

                hyper_gradient = None
                mom_gradient = None
                if index in self.hessian_hyper_gradient_mom_dict:
                    (
                        hyper_gradient,
                        mom_gradient,
                    ) = self.__get_hyper_gradient_and_momentum(
                        index, use_approximation=False
                    )

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

                assert (hyper_gradient is not None and mom_gradient is not None) or (
                    hyper_gradient is None and mom_gradient is None
                )
                if hyper_gradient is not None:
                    self.__set_hyper_gradient_and_momentum(
                        index, hyper_gradient, mom_gradient, use_approximation=False
                    )
                self.hessian_computation_arguments[index] = None
            get_logger().info(
                "__do_computation_with_hessian chunk size %s use time %s ms",
                len(chunk),
                counter.elapsed_milliseconds(),
            )

    def do_delayed_computation(self, index=None):
        if index is None:
            fast_keys = self.approx_hyper_gradient_mom_dict.in_memory_keys()
            get_logger().info(
                "begin do do_delayed_computation from fast keys %s", len(fast_keys)
            )
            for k in fast_keys:
                if (
                    k in self.delayed_approximation_computations
                    and self.delayed_approximation_computations[k]
                ):
                    self.do_delayed_computation(k)
            get_logger().info("end do do_delayed_computation from fast keys")
            self.approx_hyper_gradient_mom_dict.flush_all(wait=False)

            unfinished_keys = []
            for k, v in self.delayed_approximation_computations.items():
                if v:
                    unfinished_keys.append(k)

            for chunk in split_list_to_chunks(unfinished_keys, self.cache_size // 2):
                self.approx_hyper_gradient_mom_dict.prefetch(chunk)
                for k in chunk:
                    get_logger().info("do delayed_approximation_computations for %s", k)
                    self.do_delayed_computation(k)
            return

        hyper_gradient = None
        mom_gradient = None
        if str(index) in self.approx_hyper_gradient_mom_dict:
            hyper_gradient, mom_gradient = self.__get_hyper_gradient_and_momentum(
                index, use_approximation=True
            )

        for arguments in self.delayed_approximation_computations[index]:
            (momentum, weight_decay, learning_rate, instance_gradient) = arguments
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

        assert hyper_gradient is not None
        assert mom_gradient is not None
        self.delayed_approximation_computations[index] = []
        self.__set_hyper_gradient_and_momentum(
            index, hyper_gradient, mom_gradient, use_approximation=True
        )

    @staticmethod
    def create_hypergradient_dict(
        cache_size,
        model=None,
        storage_dir=None,
        concat_momentum=True,
    ):
        mask = None
        gradient_shape = None
        if model is not None and prune.is_pruned(model):
            model_util = ModelUtil(model)
            get_logger().info(
                "use pruned model, sparsity is %s", model_util.get_sparsity()[0]
            )
            parameters = model_util.get_parameter_list()
            gradient_shape = parameters.shape
            mask = model_util.get_pruning_mask_list()
            assert len(mask) == len(parameters)
        if mask is not None:
            if concat_momentum:
                mask = torch.cat((mask, mask))
                gradient_shape[1] *= 2
        tensor_dict = SyncedTensorDict.create(
            cache_size=cache_size,
            storage_dir=storage_dir,
            mask=mask,
            tensor_shape=gradient_shape,
        )
        return tensor_dict

    def _before_batch(self, **kwargs):
        super()._before_batch(**kwargs)
        trainer = kwargs["model_executor"]
        batch = kwargs["batch"]

        assert len(batch) == 3
        instance_info = trainer.decode_batch(batch)[2]
        instance_indices = instance_info["index"]
        instance_indices = {idx.data.item() for idx in batch[2]["index"]}

        batch_gradient_indices = instance_indices & self.computed_indices
        assert batch_gradient_indices == self.sample_gradients.keys()

        if self.use_approximation:
            self.approx_hyper_gradient_mom_dict.prefetch(
                [str(i) for i in batch_gradient_indices]
            )

        if self.use_hessian:
            self.hvp_function = get_hessian_vector_product_func(
                trainer.model_with_loss, batch
            )
            self.hessian_computation_arguments = dict()
        else:
            self.hessian_computation_arguments = None

    def _after_batch(self, **kwargs):
        trainer = kwargs["model_executor"]
        optimizer = trainer.get_optimizer()
        if not isinstance(optimizer, torch.optim.SGD):
            raise RuntimeError("optimizer is not SGD")

        cur_learning_rates = trainer.get_data("cur_learning_rates")
        assert len(cur_learning_rates) == 1
        cur_learning_rate = cur_learning_rates[0]
        batch_size = get_batch_size(kwargs.get("batch"))

        momentums = [group["momentum"] for group in optimizer.param_groups]
        if len(momentums) != 1:
            raise RuntimeError("unsupported momentums")

        momentum = momentums[0]
        weight_decay = trainer.hyper_parameter.weight_decay
        training_set_size = len(trainer.dataset)

        for idx in self.computed_indices:
            instance_gradient = None
            if idx in self.sample_gradients:
                instance_gradient = (
                    (self.sample_gradients[idx] * training_set_size / batch_size)
                    .detach()
                    .clone()
                )
            if self.use_hessian:
                self.hessian_computation_arguments[idx] = (
                    momentum,
                    weight_decay,
                    cur_learning_rate,
                    instance_gradient,
                )
            if self.use_approximation:
                self.delayed_approximation_computations[idx].append(
                    (momentum, weight_decay, cur_learning_rate, instance_gradient)
                )
        if self.use_hessian:
            self.__do_computation_with_hessian()
        if self.use_approximation:
            for idx in self.computed_indices:
                if idx in self.sample_gradients:
                    self.do_delayed_computation(idx)

    def __get_hyper_gradient_and_momentum(self, index, use_approximation):
        tmp = None
        if use_approximation:
            tmp = self.approx_hyper_gradient_mom_dict[index]
        else:
            tmp = self.hessian_hyper_gradient_mom_dict[index]
        return torch.split(tmp, tmp.shape[0] // 2)

    def foreach_hyper_gradient(self, use_approximation, callback):
        hyper_gradient_mom_dict = None
        if use_approximation:
            hyper_gradient_mom_dict = self.approx_hyper_gradient_mom_dict
        else:
            hyper_gradient_mom_dict = self.hessian_hyper_gradient_mom_dict
        for index in hyper_gradient_mom_dict.keys():
            if use_approximation:
                self.do_delayed_computation(index)
            hyper_gradient, _ = self.__get_hyper_gradient_and_momentum(
                index, use_approximation
            )
            callback(index, hyper_gradient)

    def __set_hyper_gradient_and_momentum(
        self, index, hyper_gradient, mom_gradient, use_approximation
    ):
        index = str(index)
        if use_approximation:
            self.approx_hyper_gradient_mom_dict[index] = torch.cat(
                (hyper_gradient, mom_gradient)
            )
        else:
            self.hessian_hyper_gradient_mom_dict[index] = torch.cat(
                (hyper_gradient, mom_gradient)
            )

    def get_hyper_gradient(self, index, use_approximation):
        return self.__get_hyper_gradient_and_momentum(index, use_approximation)[0]

    def __save_hyper_gradients(self, trainer, hyper_gradient_dir, use_approximation):
        if use_approximation:
            get_logger().info("begin do do_delayed_computation")
            self.do_delayed_computation()
            get_logger().info("end do do_delayed_computation")
        hyper_gradient_dict = HyperGradientCallback.create_hypergradient_dict(
            self.cache_size, trainer.model
        )
        hyper_gradient_dict.set_storage_dir(hyper_gradient_dir)

        hyper_gradient_mom_dict = (
            self.approx_hyper_gradient_mom_dict
            if use_approximation
            else self.hessian_hyper_gradient_mom_dict
        )
        for chunk in split_list_to_chunks(
            hyper_gradient_mom_dict.keys(), self.cache_size // 2
        ):
            hyper_gradient_mom_dict.prefetch(chunk)
            for index in chunk:
                hyper_gradient = self.get_hyper_gradient(index, use_approximation)
                hyper_gradient_dict[index] = hyper_gradient
        trainer.save_model(os.path.join(hyper_gradient_dir, "model"))
        hyper_gradient_dict.tensor_dict.flush_all(True)
        hyper_gradient_dict.tensor_dict.release()
        hyper_gradient_dict = None
        super()._after_execute()
