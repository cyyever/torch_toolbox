from cyy_torch_toolbox.inferencer import Inferencer

# from cyy_torch_toolbox.metrics.prob_metric import ProbabilityMetric


class ClassificationInferencer(Inferencer):
    pass
    # def inference(self, **kwargs) -> bool:
    #     sample_prob = kwargs.get("sample_prob", False)
    #     if sample_prob:
    #         if not self.has_hook_obj("probability_metric"):
    #             self.append_hook(ProbabilityMetric(), "probability_metric")
    #         self.enable_hook(hook_name="probability_metric")
    #     else:
    #         self.disable_hook(hook_name="probability_metric")
    #     return super().inference(**kwargs)
