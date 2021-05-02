import copy
import logging
from typing import Callable

import torch
from cyy_naive_lib.log import get_logger

from dataset_collection import DatasetCollection
from hooks.learning_rate_hook import LearningRateHook
from hooks.save_model import SaveModelHook
from hyper_parameter import HyperParameter
from inference import ClassificationInferencer, DetectionInferencer, Inferencer
from metric_visualizers.batch_loss_logger import BatchLossLogger
from metric_visualizers.metric_visdom import MetricVisdom
from ml_type import MachineLearningPhase, ModelType, StopExecutingException
from model_executor import ModelExecutor, ModelExecutorCallbackPoint
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
        )
        LearningRateHook().append_to_model_executor(self)
        self.__inferencers: dict = dict()
        self.__batch_loss_logger = BatchLossLogger()
        self.__batch_loss_logger.append_to_model_executor(self)
        self.__metric_visdom: MetricVisdom = MetricVisdom()
        self.__metric_visdom.append_to_model_executor(self)
        self.__save_model_hook = SaveModelHook()
        self.save_dir = save_dir

    @property
    def batch_loss_logger(self):
        return self.__batch_loss_logger

    @property
    def metric_visdom(self):
        return self.__metric_visdom

    def get_inferencer_performance_metric(self, phase):
        return self.__inferencers[phase].performance_metric

    def get_inferencer(
        self, phase: MachineLearningPhase, copy_model=False
    ) -> Inferencer:
        assert phase != MachineLearningPhase.Training
        model_with_loss = copy.copy(self.model_with_loss)
        if copy_model:
            get_logger().debug("copy model in inferencer")
            model_with_loss.set_model(copy.deepcopy(self.model))

        if self.model_with_loss.model_type == ModelType.Classification:
            return ClassificationInferencer(
                model_with_loss,
                self.dataset_collection,
                phase=phase,
                hyper_parameter=self.hyper_parameter,
            )
        if self.model_with_loss.model_type == ModelType.Detection:
            return DetectionInferencer(
                model_with_loss,
                self.dataset_collection,
                phase=phase,
                hyper_parameter=self.hyper_parameter,
                iou_threshold=0.6,
            )
        raise RuntimeError(
            "Unsupported model type:" + str(self.model_with_loss.model_type)
        )

    def get_optimizer(self):
        if not self.has_data("optimizer"):
            self.set_data(
                "optimizer",
                self.hyper_parameter.get_optimizer(self),
            )
        return self.get_data("optimizer")

    def remove_optimizer(self):
        self.remove_data("optimizer")

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

    def load_parameter_dict(self, parameter_dict: dict, remove_lr_scheduler: bool):
        ModelUtil(self.model).load_parameter_dict(parameter_dict)
        self.remove_optimizer()
        if remove_lr_scheduler:
            self.remove_lr_scheduler()

    def repeated_train(self, repeated_num, save_dir=None, **kwargs):
        def training_callback(_, trainer: Trainer):
            nonlocal save_dir, kwargs
            get_logger().setLevel(logging.ERROR)
            kwargs["test_epoch_interval"] = 1
            trainer.train(**kwargs)
            if save_dir is not None:
                trainer.save_model(save_dir)
            get_logger().setLevel(logging.DEBUG)
            return {
                "training_loss": trainer.training_loss,
                "validation_loss": trainer.validation_loss,
                "validation_accuracy": trainer.validation_accuracy,
                "test_loss": trainer.test_loss,
                "test_accuracy": trainer.test_accuracy,
            }

        return Trainer.__repeated_training(repeated_num, self, training_callback)

    def train(self, **kwargs):
        self.remove_optimizer()
        self.remove_lr_scheduler()

        if kwargs.get("save_model", True) and self.save_dir is not None:
            self.__save_model_hook.append_to_model_executor(self)
        else:
            self.__save_model_hook.remove_from_model_executor(self)
        self.__inferencers.clear()
        for phase in (MachineLearningPhase.Validation, MachineLearningPhase.Test):
            self.__inferencers[phase] = self.get_inferencer(phase)
            self.__inferencers[phase].remove_logger()
        self.exec_callbacks(
            ModelExecutorCallbackPoint.BEFORE_EXECUTE, model_executor=self
        )

        try:
            for epoch in range(1, self.hyper_parameter.epoch + 1):
                self.exec_callbacks(
                    ModelExecutorCallbackPoint.BEFORE_EPOCH,
                    model_executor=self,
                    epoch=epoch,
                )
                if self.cuda_stream is not None:
                    get_logger().debug("use cuda stream %s", self.cuda_stream)

                with torch.cuda.stream(self.cuda_stream):
                    for batch_index, batch in enumerate(self.dataloader):
                        optimizer = self.get_optimizer()
                        lr_scheduler = self.get_lr_scheduler()
                        assert optimizer is not None
                        assert lr_scheduler is not None
                        batch_size = self.get_batch_size(batch[0])
                        if (
                            self.model_with_loss.has_batch_norm
                            and self.hyper_parameter.batch_size != 1
                            and batch_size == 1
                        ):
                            get_logger().debug("drop last one-batch for batchnorm")
                            continue

                        optimizer.zero_grad()
                        self.exec_callbacks(
                            ModelExecutorCallbackPoint.BEFORE_BATCH,
                            model_executor=self,
                            batch_index=batch_index,
                            batch=batch,
                            batch_size=batch_size,
                        )
                        optimizer.zero_grad()
                        sample_inputs, sample_targets, _ = self.decode_batch(batch)
                        result = self.model_with_loss(
                            sample_inputs,
                            sample_targets,
                            phase=self.phase,
                            device=self.device,
                        )
                        loss = result["loss"]
                        loss.backward()
                        batch_loss = loss.data.item()

                        self.exec_callbacks(
                            ModelExecutorCallbackPoint.AFTER_BATCH,
                            model_executor=self,
                            batch_index=batch_index,
                            batch=batch,
                            epoch=epoch,
                            result=result,
                            batch_loss=batch_loss,
                            batch_size=batch_size,
                        )
                        if self.has_callback(ModelExecutorCallbackPoint.OPTIMIZER_STEP):
                            self.exec_callbacks(
                                ModelExecutorCallbackPoint.OPTIMIZER_STEP,
                                self,
                            )
                        else:
                            optimizer.step()

                        optimizer.zero_grad()
                        self.exec_callbacks(
                            ModelExecutorCallbackPoint.AFTER_OPTIMIZER_STEP,
                            model_executor=self,
                            batch_index=batch_index,
                            batch=batch,
                            epoch=epoch,
                            result=result,
                            batch_loss=batch_loss,
                            batch_size=batch_size,
                        )

                        if HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                            get_logger().debug("adjust lr after batch")
                            lr_scheduler.step()

                # update model parameters
                for inferencer in self.__inferencers.values():
                    inferencer.set_model(copy.deepcopy(self.model))
                    inferencer.inference(epoch=epoch, use_grad=False)

                self.exec_callbacks(
                    ModelExecutorCallbackPoint.AFTER_EPOCH,
                    model_executor=self,
                    epoch=epoch,
                )

                if not HyperParameter.lr_scheduler_step_after_batch(lr_scheduler):
                    if isinstance(
                        lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        training_loss = self.performance_metric.get_loss(epoch)
                        get_logger().debug(
                            "call ReduceLROnPlateau for training loss %s", training_loss
                        )
                        lr_scheduler.step(training_loss)
                    else:
                        lr_scheduler.step()
        except StopExecutingException:
            get_logger().warning("stop training")
        self.exec_callbacks(
            ModelExecutorCallbackPoint.AFTER_EXECUTE, model_executor=self
        )

    @staticmethod
    def __repeated_training(number: int, trainer, training_callback: Callable):
        results: dict = dict()
        for idx in range(number):
            statistics = training_callback(idx, trainer)
            assert isinstance(statistics, dict)
            for k, v in statistics.items():
                tensor = None
                if isinstance(v, list):
                    tensor = torch.Tensor(v)
                elif isinstance(v, dict):
                    tensor = torch.Tensor([v[k] for k in sorted(v.keys())])
                else:
                    raise RuntimeError("unsupported value" + str(v))
                if k in results:
                    results[k] += tensor
                else:
                    results[k] = tensor
        for k, v in results.items():
            results[k] = v / number
        return results
