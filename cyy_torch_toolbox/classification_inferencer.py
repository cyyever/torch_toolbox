from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.metrics.prob_metric import ProbabilityMetric

# from torchvision.ops.boxes import box_iou


class ClassificationInferencer(Inferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__prob_metric = None

    @property
    def prob_metric(self) -> ProbabilityMetric:
        return self.__prob_metric

    def inference(self, **kwargs):
        sample_prob = kwargs.get("sample_prob", False)
        if sample_prob:
            if self.__prob_metric is None:
                self.__prob_metric = ProbabilityMetric()
                self.append_hook(self.__prob_metric)
        else:
            if self.__prob_metric is not None:
                self.disable_hook(self.__prob_metric)
        super().inference(**kwargs)


# class DetectionInferencer(Inferencer):
#     def __init__(self, *args, **kwargs):
#         iou_threshold = kwargs.pop("iou_threshold", None)
#         assert iou_threshold
#         super().__init__(*args, **kwargs)
#         self.iou_threshold = iou_threshold

#     def inference(self, **kwargs):
#         detection_count_per_label = dict()
#         detection_correct_count_per_label = dict()

#         dataset_util = DatasetUtil(self.dataset)
#         for label in dataset_util.get_labels():
#             detection_correct_count_per_label[label] = 0
#             detection_count_per_label[label] = 0

#         def after_batch_hook(_, batch, result):
#             targets = batch[1]
#             for target in targets:
#                 for label in target["labels"]:
#                     label = label.data.item()
#                     detection_count_per_label[label] += 1

#             detection: list = result["detection"]
#             for idx, sample_detection in enumerate(detection):
#                 detected_boxex = sample_detection["boxes"]
#                 if detected_boxex.nelement() == 0:
#                     continue
#                 target = targets[idx]
#                 target_boxex = target["boxes"]
#                 iou_matrix = box_iou(target_boxex, detected_boxex)
#                 for box_idx, iou in enumerate(iou_matrix):
#                     max_iou_index = torch.argmax(iou)
#                     if iou[max_iou_index] < self.iou_threshold:
#                         continue
#                     print(
#                         "max iou is",
#                         iou[max_iou_index],
#                         "target label",
#                         target["labels"][box_idx],
#                         "detected label",
#                         sample_detection["labels"][max_iou_index],
#                     )
#                     if (
#                         target["labels"][box_idx]
#                         == sample_detection["labels"][max_iou_index]
#                     ):
#                         label = target["labels"][box_idx].data.item()
#                         detection_correct_count_per_label[label] += 1

#         self.append_hook(
#             ModelExecutorHookPoint.AFTER_BATCH, "compute_acc", after_batch_hook
#         )
#         super().inference(**kwargs)
#         self.remove_hook("compute_acc", ModelExecutorHookPoint.AFTER_BATCH)

#         accuracy = sum(detection_correct_count_per_label.values()) / sum(
#             detection_count_per_label.values()
#         )
#         per_class_accuracy = dict()
#         for label in detection_count_per_label:
#             per_class_accuracy[label] = (
#                 detection_correct_count_per_label[label]
#                 / detection_count_per_label[label]
#             )
#         return (self.loss, accuracy, {"per_class_accuracy": per_class_accuracy})
