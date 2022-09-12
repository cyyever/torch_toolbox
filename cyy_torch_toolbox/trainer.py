import torch
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.classification_inferencer import \
    ClassificationInferencer
from cyy_torch_toolbox.dataset import get_dataset_size
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.hooks.keep_model import KeepModelHook
from cyy_torch_toolbox.hooks.learning_rate_hook import LearningRateHook
from cyy_torch_toolbox.hooks.trainer_debugger import TrainerDebugger
from cyy_torch_toolbox.hyper_parameter import HyperParameter
from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.metric_visualizers.batch_loss_logger import \
    BatchLossLogger
from cyy_torch_toolbox.ml_type import (MachineLearningPhase,
                                       ModelExecutorHookPoint, ModelType,
                                       StopExecutingException)
from cyy_torch_toolbox.model_executor import ModelExecutor
from cyy_torch_toolbox.model_with_loss import (ModelWithLoss,
                                               ParallelModelWithLoss)


class Trainer(ModelExecutor):
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset_collection: DatasetCollection,
        hyper_parameter: HyperParameter,
    ):
        super().__init__(
            model_with_loss,
            dataset_collection,
            MachineLearningPhase.Training,
            hyper_parameter,
        )
        self.append_hook(LearningRateHook())
        self.__inferencers: dict = {}
        self.__batch_loss_logger = BatchLossLogger()
        self.append_hook(self.__batch_loss_logger)
        self.__keep_model_hook = KeepModelHook()
        self.append_hook(self.__keep_model_hook)
        self.__debugger = None

    def set_device(self, device):
        super().set_device(device)
        for inferencer in self.__inferencers.values():
            inferencer.set_device(device)

    @property
    def batch_loss_logger(self):
        return self.__batch_loss_logger

    @property
    def best_model(self):
        if self.__keep_model_hook.best_model is None:
            return None
        return self.__keep_model_hook.best_model[0]

    def get_inferencer_performance_metric(self, phase):
        return self.__inferencers[phase].performance_metric

    def get_inferencer(
        self, phase: MachineLearningPhase, copy_model: bool = False
    ) -> Inferencer:
        model_with_loss = self.copy_model_with_loss(deepcopy=copy_model)

        inferencer = None
        if model_with_loss.model_type == ModelType.Classification:
            inferencer = ClassificationInferencer(
                model_with_loss,
                self.dataset_collection,
                phase=phase,
                hyper_parameter=self.hyper_parameter,
            )
        if inferencer is None:
            raise RuntimeError(
                "Unsupported model type:" + str(model_with_loss.model_type)
            )
        inferencer.cache_transforms = self.cache_transforms
        inferencer.set_device(self.device)
        if self.has_amp():
            inferencer.set_amp()
        return inferencer

    def get_optimizer(self):
        if not self.has_data("optimizer"):
            self.set_data(
                "optimizer",
                self.hyper_parameter.get_optimizer(self),
            )
        return self.get_data("optimizer")

    def remove_optimizer(self):
        self.remove_data("optimizer")
        self.remove_lr_scheduler()

    def get_lr_scheduler(self):
        if not self.has_data("lr_scheduler"):
            self.set_data(
                "lr_scheduler",
                self.hyper_parameter.get_lr_scheduler(self),
            )
        return self.get_data("lr_scheduler")

    def remove_lr_scheduler(self):
        self.remove_data("lr_scheduler")

    def load_model(self, model_path):
        super().load_model(model_path)
        self.remove_optimizer()

    def load_parameter_dict(self, parameter_dict: dict) -> None:
        self.model_util.load_parameter_dict(parameter_dict)
        self.remove_optimizer()

    def offload_from_gpu(self):
        self.__inferencers.clear()
        super().offload_from_gpu()

    def offload_from_memory(self):
        super().offload_from_memory()
        self.__keep_model_hook.offload_from_memory()

    def load_to_memory(self):
        super().load_to_memory()

    def add_skipped_epoch(self, epoch):
        key = "skipped_epoch"
        old_data = self.get_data(key, set())
        old_data.add(epoch)
        self.set_data(key, old_data)

    def _prepare_execution(
        self,
        use_DDP: bool = False,
        save_best_model: bool = False,
        save_epoch_model: bool = False,
        **kwargs
    ):
        self.__keep_model_hook.save_best_model = save_best_model
        self.__keep_model_hook.save_epoch_model = save_epoch_model
        self.__inferencers.clear()
        if self.debugging_mode:
            get_logger().warning("train in debugging mode")
            if self.__debugger is None:
                self.__debugger = TrainerDebugger()
                self.append_hook(self.__debugger)
            else:
                self.enable_hook(self.__debugger)
        else:
            if self.__debugger is not None:
                self.disable_hook(self.__debugger)

        if use_DDP:
            self._model_with_loss = ParallelModelWithLoss.create(self._model_with_loss)
        super()._prepare_execution(**kwargs)
        for phase in MachineLearningPhase:
            if self.dataset_collection.has_dataset(phase):
                get_logger().info(
                    "%s dataset len %s",
                    phase,
                    get_dataset_size(self.dataset_collection.get_dataset(phase=phase)),
                )
        self.exec_hooks(ModelExecutorHookPoint.BEFORE_EXECUTE)

    def train(self, **kwargs):
        self._prepare_execution(**kwargs)

        with torch.cuda.stream(self.cuda_stream):
            try:
                for epoch in range(1, self.hyper_parameter.epoch + 1):
                    self._execute_epoch(
                        epoch=epoch, need_backward=True, in_training=True
                    )

                    for phase in (
                        MachineLearningPhase.Validation,
                        MachineLearningPhase.Test,
                    ):
                        if (
                            phase not in self.__inferencers
                            and self.dataset_collection.has_dataset(phase=phase)
                        ):
                            inferencer = self.get_inferencer(phase)
                            inferencer.disable_logger()
                            self.__inferencers[phase] = inferencer
                        inferencer = self.__inferencers.get(phase, None)
                        if inferencer is not None:
                            inferencer.model.load_state_dict(self.model.state_dict())
                            inferencer.inference(epoch=epoch, use_grad=False)

                    self.exec_hooks(
                        ModelExecutorHookPoint.AFTER_VALIDATION,
                        epoch=epoch,
                    )
                    # adjust learning rate
                    lr_scheduler = self.get_lr_scheduler()
                    if HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                        continue
                    match lr_scheduler:
                        case torch.optim.lr_scheduler.ReduceLROnPlateau():
                            training_loss = self.performance_metric.get_loss(epoch)
                            get_logger().debug(
                                "call ReduceLROnPlateau for training loss %s",
                                training_loss,
                            )
                            lr_scheduler.step(training_loss)
                        case _:
                            lr_scheduler.step()
            except StopExecutingException:
                get_logger().warning("stop training")
            finally:
                self._wait_stream()
        self.exec_hooks(
            ModelExecutorHookPoint.AFTER_EXECUTE,
            epoch=self.hyper_parameter.epoch,
        )

    def _get_backward_loss(self, result):
        return result["loss"]
