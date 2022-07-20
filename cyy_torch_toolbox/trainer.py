import torch
from cyy_naive_lib.log import get_logger

from cyy_torch_toolbox.classification_inferencer import \
    ClassificationInferencer
from cyy_torch_toolbox.dataset import get_dataset_size
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.hooks.amp import AMP

try:
    from cyy_torch_toolbox.hooks.amp import ApexAMP

    has_apex_amp = True
except BaseException:
    has_apex_amp = False

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
        self.__amp_hook = None

    def set_device(self, device):
        super().set_device(device)
        for a in self.__inferencers.values():
            a.set_device(device)

    def set_amp(self, enabled=True, use_apex=False):
        if self.__amp_hook is not None:
            self.remove_hook(self.__amp_hook)
            self.__amp_hook = None
        if enabled:
            if use_apex:
                assert has_apex_amp
                self.__amp_hook = ApexAMP()
            else:
                self.__amp_hook = AMP()
            self.append_hook(self.__amp_hook)

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
        inferencer.set_device(self.device)
        return inferencer

    def get_optimizer(self):
        if not self.has_data("optimizer"):
            self.set_data(
                "optimizer",
                self.hyper_parameter.get_optimizer(self),
            )
        return self.get_data("optimizer")

    def remove_optimizer(self):
        # Don't call this method until you are sure what you are doing in federated learning settings.
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
        self.__keep_model_hook.offload_from_memory(self.save_dir)

    def load_to_memory(self):
        super().load_to_memory()
        self.__keep_model_hook.load_to_memory(self.save_dir)

    def add_skipped_epoch(self, epoch):
        key = "skipped_epoch"
        old_data = self.get_data(key, set())
        old_data.add(epoch)
        self.set_data(key, old_data)

    def _prepare_execution(self, use_DDP=False, save_model=False, **kwargs):
        self.__keep_model_hook.save_flag = save_model and self.save_dir is not None
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
                    if epoch in self.get_data("skipped_epoch", set()):
                        get_logger().warning("skip epoch %s", epoch)
                        continue
                    self.exec_hooks(
                        ModelExecutorHookPoint.BEFORE_EPOCH,
                        epoch=epoch,
                    )
                    self.exec_hooks(
                        ModelExecutorHookPoint.BEFORE_FETCH_BATCH, batch_index=0
                    )
                    for batch_index, batch in enumerate(self.dataloader):
                        self.exec_hooks(
                            ModelExecutorHookPoint.AFTER_FETCH_BATCH,
                            batch_index=batch_index,
                        )

                        optimizer = self.get_optimizer()
                        optimizer.zero_grad(set_to_none=True)

                        (
                            batch_size,
                            sample_inputs,
                            sample_targets,
                            other_info,
                        ) = self.decode_batch(batch)
                        if batch_size is None:
                            batch_size = self.get_batch_size(sample_targets)
                        batch = (sample_inputs, sample_targets, other_info)
                        if (
                            self.hyper_parameter.batch_size != 1
                            and batch_size == 1
                            and self._model_with_loss.has_batch_norm
                        ):
                            get_logger().debug("drop last one-batch for batchnorm")
                            continue

                        self.exec_hooks(
                            ModelExecutorHookPoint.BEFORE_BATCH,
                            batch_index=batch_index,
                            batch=batch,
                            batch_size=batch_size,
                        )
                        kwargs = {
                            "inputs": sample_inputs,
                            "targets": sample_targets,
                            "phase": self.phase,
                            "device": self.device,
                            "non_blocking": True,
                        }
                        if self.has_hook(ModelExecutorHookPoint.MODEL_FORWARD):
                            self.exec_hooks(
                                ModelExecutorHookPoint.MODEL_FORWARD,
                                model_kwargs=kwargs,
                            )
                            result = self.get_data("forward_result")
                        else:
                            result = self._model_with_loss(**kwargs)
                        self.exec_hooks(
                            ModelExecutorHookPoint.BEFORE_BACKWARD,
                            inputs=result["inputs"],
                            input_features=result["input_features"],
                            targets=result["targets"],
                            batch_info=other_info,
                        )
                        if self.has_hook(ModelExecutorHookPoint.MODEL_BACKWARD):
                            self.exec_hooks(
                                ModelExecutorHookPoint.MODEL_BACKWARD,
                                loss=result["loss"],
                            )
                        else:
                            result["loss"].backward()

                        self.exec_hooks(
                            ModelExecutorHookPoint.AFTER_BATCH,
                            batch_index=batch_index,
                            inputs=result["inputs"],
                            input_features=result["input_features"],
                            targets=result["targets"],
                            batch_info=other_info,
                            epoch=epoch,
                            result=result,
                            batch_size=batch_size,
                        )
                        if self.has_hook(ModelExecutorHookPoint.OPTIMIZER_STEP):
                            self.exec_hooks(ModelExecutorHookPoint.OPTIMIZER_STEP)
                        else:
                            optimizer.step()
                        lr_scheduler = self.get_lr_scheduler()
                        if HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                            get_logger().debug("adjust lr after batch")
                            lr_scheduler.step()

                        self.exec_hooks(
                            ModelExecutorHookPoint.AFTER_OPTIMIZER_STEP,
                            epoch=epoch,
                            batch_index=batch_index,
                            batch=batch,
                            batch_size=batch_size,
                        )

                        self.exec_hooks(
                            ModelExecutorHookPoint.BEFORE_FETCH_BATCH,
                            batch_index=batch_index + 1,
                        )

                    self.exec_hooks(
                        ModelExecutorHookPoint.AFTER_EPOCH,
                        epoch=epoch,
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
