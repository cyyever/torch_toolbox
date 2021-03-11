import copy

import torch
from cyy_naive_lib.log import get_logger
from torchvision.ops.boxes import box_iou

from dataset import DatasetUtil
from dataset_collection import DatasetCollection
from hyper_parameter import HyperParameter
from metrics.acc_metric import AccuracyMetric
from metrics.loss_metric import LossMetric
from metrics.prob_metric import ProbabilityMetric
from ml_type import MachineLearningPhase
from model_executor import ModelExecutor, ModelExecutorCallbackPoint
from model_util import ModelUtil
from model_with_loss import ModelWithLoss


class Inferencer(ModelExecutor):
    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        dataset_collection: DatasetCollection,
        phase: MachineLearningPhase,
        hyper_parameter: HyperParameter,
        copy_model=True,
    ):
        assert phase != MachineLearningPhase.Training
        super().__init__(model_with_loss, dataset_collection, phase, hyper_parameter)
        if copy_model:
            get_logger().debug("copy model in inferencer")
            self.model_with_loss.set_model(copy.deepcopy(model_with_loss.model))
        self._loss_metric = LossMetric(self)

    @property
    def loss(self):
        return self._loss_metric.get_loss(1)

    def inference(self, **kwargs):
        self.set_data("dataset_size", len(self.dataset))
        use_grad = kwargs.get("use_grad", False)
        with torch.set_grad_enabled(use_grad):
            get_logger().debug("use device %s", self.device)
            self.model.zero_grad()
            self.model.to(self.device)
            self.exec_callbacks(
                ModelExecutorCallbackPoint.BEFORE_EPOCH,
                self,
                1,
            )
            for batch_index, batch in enumerate(self.dataloader):
                inputs, targets, _ = self.decode_batch(batch)
                result = self.model_with_loss(inputs, targets, phase=self.phase)
                batch_loss = result["loss"]
                if use_grad:
                    batch_loss.backward()

                self.exec_callbacks(
                    ModelExecutorCallbackPoint.AFTER_BATCH,
                    self,
                    batch=batch,
                    batch_loss=batch_loss,
                    batch_index=batch_index,
                    result=result,
                    epoch=1,
                )
            self.exec_callbacks(
                ModelExecutorCallbackPoint.AFTER_EPOCH,
                self,
                1,
            )
            return

    def get_gradient(self):
        self.inference(use_grad=True)
        return ModelUtil(self.model).get_gradient_list()


class ClassificationInferencer(Inferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__acc_metric = AccuracyMetric(self)
        self.__prob_metric = None

    @property
    def accuracy_metric(self) -> AccuracyMetric:
        return self.__acc_metric

    def inference(self, *args, **kwargs):
        per_sample_prob = kwargs.get("per_sample_prob", False)
        if per_sample_prob:
            self.__prob_metric = ProbabilityMetric(self)
        else:
            if self.__prob_metric is not None:
                self.__prob_metric.remove_callbacks()
        super().inference(*args, **kwargs)


class DetectionInferencer(Inferencer):
    def __init__(self, *args, **kwargs):
        iou_threshold = kwargs.pop("iou_threshold", None)
        assert iou_threshold
        super().__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold

    def inference(self, **kwargs):
        detection_count_per_label = dict()
        detection_correct_count_per_label = dict()

        dataset_util = DatasetUtil(self.dataset)
        for label in dataset_util.get_labels():
            detection_correct_count_per_label[label] = 0
            detection_count_per_label[label] = 0

        def after_batch_callback(_, batch, result):
            targets = batch[1]
            for target in targets:
                for label in target["labels"]:
                    label = label.data.item()
                    detection_count_per_label[label] += 1

            detection: list = result["detection"]
            for idx, sample_detection in enumerate(detection):
                detected_boxex = sample_detection["boxes"]
                if detected_boxex.nelement() == 0:
                    continue
                target = targets[idx]
                target_boxex = target["boxes"]
                iou_matrix = box_iou(target_boxex, detected_boxex)
                for box_idx, iou in enumerate(iou_matrix):
                    max_iou_index = torch.argmax(iou)
                    if iou[max_iou_index] < self.iou_threshold:
                        continue
                    print(
                        "max iou is",
                        iou[max_iou_index],
                        "target label",
                        target["labels"][box_idx],
                        "detected label",
                        sample_detection["labels"][max_iou_index],
                    )
                    if (
                        target["labels"][box_idx]
                        == sample_detection["labels"][max_iou_index]
                    ):
                        label = target["labels"][box_idx].data.item()
                        detection_correct_count_per_label[label] += 1

        self.add_named_callback(
            ModelExecutorCallbackPoint.AFTER_BATCH, "compute_acc", after_batch_callback
        )
        super().inference(**kwargs)
        loss = self.loss
        self.remove_callback("compute_acc", ModelExecutorCallbackPoint.AFTER_BATCH)

        accuracy = sum(detection_correct_count_per_label.values()) / sum(
            detection_count_per_label.values()
        )
        per_class_accuracy = dict()
        for label in detection_count_per_label:
            per_class_accuracy[label] = (
                detection_correct_count_per_label[label]
                / detection_count_per_label[label]
            )
        return (loss, accuracy, {"per_class_accuracy": per_class_accuracy})
