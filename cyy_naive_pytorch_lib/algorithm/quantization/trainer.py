import copy
from typing import Callable, Optional

import torch
from cyy_naive_lib.log import get_logger

from hyper_parameter import HyperParameter
from model_loss import ModelWithLoss
from model_util import ModelUtil
from trainer import Trainer


class QuantizationTrainer(Trainer):
    """
    This trainer is used for Training Aware Quantization
    """

    def __init__(
        self,
        model_with_loss: ModelWithLoss,
        training_dataset,
        hyper_parameter: Optional[HyperParameter],
        replace_layer=True,
    ):
        super().__init__(model_with_loss, training_dataset, hyper_parameter)
        if replace_layer:
            model_util = ModelUtil(copy.deepcopy(self.model))
            # change ReLU6 to ReLU
            if model_util.has_sub_module(torch.nn.modules.activation.ReLU6):
                get_logger().info(
                    "replace torch.nn.modules.activation.ReLU6 to torch.nn.modules.activation.ReLU"
                )
                model_util.change_sub_modules(
                    torch.nn.modules.activation.ReLU6,
                    lambda name, sub_module: torch.nn.modules.activation.ReLU(
                        inplace=sub_module.inplace
                    ),
                )

        self.original_model = self.model
        self.quantized_model = None

    def train(self, **kwargs):
        pass

    def __prepare_quantization(self):
        if ModelUtil(self.original_model).has_sub_module(torch.quantization.QuantStub):
            quant_model = copy.deepcopy(self.original_model)
        else:
            quant_model = torch.quantization.QuantWrapper(
                copy.deepcopy(self.original_model)
            )
        quant_model.cpu()
        quant_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

        if hasattr(quant_model, "fuse_model"):
            get_logger().debug("use fuse_model of %s", type(quant_model))
            quant_model.fuse_model()
        else:
            torch.quantization.fuse_modules(
                quant_model,
                self.__get_fused_modules(quant_model),
                inplace=True,
            )
        torch.quantization.prepare_qat(quant_model, inplace=True)
        get_logger().debug("quant_model is %s", quant_model)
        self.set_model(quant_model)

    def get_quantized_model(self) -> torch.nn.Module:
        if self.quantized_model is None:
            self.model.cpu()
            self.model.eval()
            self.quantized_model = torch.quantization.convert(self.model)
        return self.quantized_model
