import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter

from classification_inferencer import ClassificationInferencer
from dataset import decode_batch
from dataset_collection import DatasetCollection
from hooks.learning_rate_hook import LearningRateHook
from hooks.save_model import SaveModelHook
from hooks.trainer_debugger import TrainerDebugger
from hyper_parameter import HyperParameter
from inference import Inferencer
from metric_visualizers.batch_loss_logger import BatchLossLogger
from ml_type import MachineLearningPhase, ModelType, StopExecutingException
from model_executor import ModelExecutor, ModelExecutorHookPoint
from model_util import ModelUtil
from model_with_loss import ModelWithLoss


class Trainer(ModelExecutor):
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset_collection: DatasetCollection,
        hyper_parameter: HyperParameter,
        save_dir=None,
    ):
        super().__init__(
            model_with_loss,
            dataset_collection,
            MachineLearningPhase.Training,
            hyper_parameter,
            save_dir=save_dir,
        )
        self.append_hook(LearningRateHook())
        self.__inferencers: dict = dict()
        self.__batch_loss_logger = BatchLossLogger()
        self.append_hook(self.__batch_loss_logger)
        self.__save_model_hook = SaveModelHook()
        self.append_hook(self.__save_model_hook)
        self.append_hook(self.visualizer)
        self.__debugger = None

    def set_device(self, device):
        super().set_device(device)
        for a in self.__inferencers.values():
            a.set_device(device)

    def use_checkpoint(self):
        self._model_with_loss.use_checkpoint = True

    @property
    def batch_loss_logger(self):
        return self.__batch_loss_logger

    def get_inferencer_performance_metric(self, phase):
        return self.__inferencers[phase].performance_metric

    def get_inferencer(
        self, phase: MachineLearningPhase, copy_model=False
    ) -> Inferencer:
        assert phase != MachineLearningPhase.Training
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

    def load_parameter_dict(self, parameter_dict: dict):
        ModelUtil(self.model).load_parameter_dict(parameter_dict)
        self.remove_optimizer()

    def offload_from_gpu(self):
        super().offload_from_gpu()
        for inferencer in self.__inferencers.values():
            inferencer.offload_from_gpu()

    def add_skipped_epoch(self, epoch):
        key = "skipped_epoch"
        old_data = self.get_data(key, set())
        old_data.add(epoch)
        self.set_data(key, old_data)

    def _prepare_execution(self, **kwargs):
        super()._prepare_execution(**kwargs)
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

        self.disable_hook(self.__save_model_hook)
        if kwargs.get("save_model", True) and self.save_dir is not None:
            self.enable_hook(self.__save_model_hook)
        self.__inferencers.clear()
        self.exec_hooks(ModelExecutorHookPoint.BEFORE_EXECUTE, model_executor=self)

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
                        model_executor=self,
                        epoch=epoch,
                    )
                    if self.profiling_mode:
                        dataloader_time_counter = TimeCounter()
                    for batch_index, batch in enumerate(self.dataloader):
                        if self.profiling_mode:
                            get_logger().warning(
                                "fetching batch used %sms",
                                dataloader_time_counter.elapsed_milliseconds(),
                            )

                        optimizer = self.get_optimizer()
                        lr_scheduler = self.get_lr_scheduler()

                        sample_inputs, sample_targets, other_info = decode_batch(batch)
                        sample_inputs = sample_inputs.to(self.device, non_blocking=True)
                        sample_targets = sample_targets.to(
                            self.device, non_blocking=True
                        )
                        batch = (sample_inputs, sample_targets, other_info)
                        batch_size = self.get_batch_size(sample_targets)
                        if (
                            self._model_with_loss.has_batch_norm
                            and self.hyper_parameter.batch_size != 1
                            and batch_size == 1
                        ):
                            get_logger().debug("drop last one-batch for batchnorm")
                            continue

                        self.exec_hooks(
                            ModelExecutorHookPoint.BEFORE_BATCH,
                            model_executor=self,
                            batch_index=batch_index,
                            batch=batch,
                            batch_size=batch_size,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        result = self._model_with_loss(
                            sample_inputs,
                            sample_targets,
                            phase=self.phase,
                            device=self.device,
                            non_blocking=True,
                        )
                        loss = result["loss"]
                        loss.backward()

                        self.exec_hooks(
                            ModelExecutorHookPoint.AFTER_BATCH,
                            model_executor=self,
                            batch_index=batch_index,
                            batch=batch,
                            epoch=epoch,
                            result=result,
                            batch_loss=loss,
                            normalized_batch_loss=result["normalized_loss"],
                            batch_size=batch_size,
                        )
                        if self.has_hook(ModelExecutorHookPoint.OPTIMIZER_STEP):
                            self.exec_hooks(
                                ModelExecutorHookPoint.OPTIMIZER_STEP,
                                model_executor=self,
                            )
                        else:
                            optimizer.step()

                        self.exec_hooks(
                            ModelExecutorHookPoint.AFTER_OPTIMIZER_STEP,
                            model_executor=self,
                            epoch=epoch,
                            batch_index=batch_index,
                            batch=batch,
                            batch_size=batch_size,
                        )

                        if HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                            get_logger().debug("adjust lr after batch")
                            lr_scheduler.step()

                        if self.profiling_mode:
                            dataloader_time_counter.reset_start_time()

                    if not self.__inferencers:
                        for phase in (
                            MachineLearningPhase.Validation,
                            MachineLearningPhase.Test,
                        ):
                            inferencer = self.get_inferencer(phase)
                            # inferencer.disable_logger()
                            inferencer.set_device(self.device)
                            self.__inferencers[phase] = inferencer

                    for inferencer in self.__inferencers.values():
                        inferencer.model.load_state_dict(self.model.state_dict())
                        inferencer.inference(epoch=epoch, use_grad=False)

                    self.exec_hooks(
                        ModelExecutorHookPoint.AFTER_EPOCH,
                        model_executor=self,
                        epoch=epoch,
                    )

                    # update model parameters
                    if not HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                        if isinstance(
                            lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            training_loss = self.performance_metric.get_loss(epoch)
                            get_logger().debug(
                                "call ReduceLROnPlateau for training loss %s",
                                training_loss,
                            )
                            lr_scheduler.step(training_loss)
                        else:
                            lr_scheduler.step()
            except StopExecutingException:
                get_logger().warning("stop training")
            self._wait_stream()
        self.exec_hooks(
            ModelExecutorHookPoint.AFTER_EXECUTE,
            model_executor=self,
            epoch=self.hyper_parameter.epoch,
        )
