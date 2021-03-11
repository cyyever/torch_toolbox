import copy

import torch
import torch.nn as nn
from cyy_naive_lib.log import get_logger
from torchvision.ops.boxes import box_iou

from dataset import DatasetUtil
from dataset_collection import DatasetCollection
from device import get_cpu_device, put_data_to_device
from hyper_parameter import HyperParameter
from metric import LossMetric
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
        super().__init__(model_with_loss, dataset_collection, hyper_parameter)
        if copy_model:
            get_logger().debug("copy model in inferencer")
            self.model_with_loss.set_model(copy.deepcopy(model_with_loss.model))
        self.__phase = phase
        self.__loss_metric = LossMetric(self)

    @property
    def dataset(self):
        return self.dataset_collection.get_dataset(phase=self.__phase)

    @property
    def loss(self):
        return self.__loss_metric.loss

    def inference(self, **kwargs):
        self.set_data("dataset_size", len(self.dataset))
        use_grad = kwargs.get("use_grad", False)
        with torch.set_grad_enabled(use_grad):
            get_logger().debug("use device %s", self.device)
            self.model_with_loss.set_model_mode(self.__phase)
            self.model.zero_grad()
            self.model.to(self.device)
            # total_loss = torch.zeros(1)
            # total_loss = total_loss.to(self.device)
            self.exec_callbacks(
                ModelExecutorCallbackPoint.BEFORE_EPOCH,
                self,
                1,
            )
            for batch_index, batch in enumerate(
                self.dataset_collection.get_dataloader(
                    self.__phase,
                    self.hyper_parameter,
                )
            ):
                inputs, targets, _ = self.decode_batch(batch)
                # real_batch_size = ModelExecutor.get_batch_size(inputs)

                result = self.model_with_loss(inputs, targets, phase=self.__phase)
                batch_loss = result["loss"]
                if use_grad:
                    batch_loss.backward()

                self.exec_callbacks(
                    ModelExecutorCallbackPoint.AFTER_BATCH,
                    batch=batch,
                    batch_loss=batch_loss,
                    batch_index=batch_index,
                    epoch=1,
                )

                # normalized_batch_loss = batch_loss
                # if self.model_with_loss.is_averaged_loss():
                #     normalized_batch_loss *= real_batch_size
                # normalized_batch_loss /= len(self.dataset)
                # if use_grad:
                #     normalized_batch_loss.backward()
                # # total_loss += normalized_batch_loss
            # return total_loss
            return

    def get_gradient(self):
        self.inference(use_grad=True)
        return ModelUtil(self.model).get_gradient_list()


class ClassificationInferencer(Inferencer):
    def inference(self, **kwargs):
        classification_count_per_label = dict()
        classification_correct_count_per_label = dict()
        sample_output = dict()
        sample_prob = dict()
        per_sample_prob = kwargs.get("per_sample_prob", False)

        dataset_util = DatasetUtil(self.dataset)
        for label in dataset_util.get_labels():
            classification_correct_count_per_label[label] = 0
            classification_count_per_label[label] = 0

        def after_batch_callback(batch, result):
            nonlocal per_sample_prob
            nonlocal sample_output
            targets = batch[1]
            output = put_data_to_device(result["output"], get_cpu_device())
            for target in targets:
                label = DatasetUtil.get_label_from_target(target)
                classification_count_per_label[label] += 1
            if per_sample_prob:
                for i, sample_index in enumerate(batch[2]):
                    sample_index = sample_index.data.item()
                    sample_output[sample_index] = output[i]
            correct = torch.eq(torch.max(output, dim=1)[1], targets).view(-1)

            for label in classification_correct_count_per_label:
                classification_correct_count_per_label[label] += torch.sum(
                    correct[targets == label]
                ).item()

        self.add_named_callback(
            ModelExecutorCallbackPoint.AFTER_BATCH, "compute_acc", after_batch_callback
        )
        super().inference(**kwargs)
        loss = self.loss
        self.remove_callback("compute_acc", ModelExecutorCallbackPoint.AFTER_BATCH)

        if per_sample_prob:
            last_layer = list(self.model.modules())[-1]
            if isinstance(last_layer, nn.LogSoftmax):
                for k, v in sample_output.items():
                    probs = torch.exp(v)
                    max_prob_index = torch.argmax(probs).data.item()
                    sample_prob[k] = (
                        max_prob_index,
                        probs[max_prob_index].data.item(),
                    )
            elif isinstance(last_layer, nn.Linear):
                for k, v in sample_output.items():
                    prob_v = nn.Softmax()(v)
                    max_prob_index = torch.argmax(prob_v).data.item()
                    sample_prob[k] = (
                        max_prob_index,
                        prob_v[max_prob_index].data.item(),
                    )
            else:
                raise RuntimeError("unsupported layer", type(last_layer))
        accuracy = sum(classification_correct_count_per_label.values()) / sum(
            classification_count_per_label.values()
        )
        per_class_accuracy = dict()
        for label in classification_count_per_label:
            per_class_accuracy[label] = (
                classification_correct_count_per_label[label]
                / classification_count_per_label[label]
            )
        return (
            loss,
            accuracy,
            {
                "per_class_accuracy": per_class_accuracy,
                "per_sample_prob": sample_prob,
            },
        )


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

        def after_batch_callback(batch, result):
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
