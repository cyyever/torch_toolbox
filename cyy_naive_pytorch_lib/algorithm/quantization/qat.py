import copy

import torch
from cyy_naive_lib.log import get_logger
from torch.quantization.fuser_method_mappings import \
    DEFAULT_OP_LIST_TO_FUSER_METHOD

from callback import Callback
from model_util import ModelUtil
from trainer import Trainer


class QuantizationAwareTraining(Callback):
    """
    Quantization-aware training
    """

    def __init__(
        self,
        replace_layer=True,
    ):
        super().__init__()
        self.__replace_layer = replace_layer
        self.__original_model = None
        self.__replace_model = None
        self.__quantized_model = None

    def _before_execute(self, **kwargs):
        trainer = kwargs["model_executor"]
        self.prepare_quantization(trainer)

    def _after_execute(self, **kwargs):
        trainer = kwargs["model_executor"]
        trainer.model.cpu()
        trainer.model.eval()
        self.__quantized_model = torch.quantization.convert(trainer.model)

    def prepare_quantization(self, trainer: Trainer):
        self.__original_model = trainer.model
        if self.__replace_layer:
            model_util = ModelUtil(copy.deepcopy(self.__original_model))
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
        else:
            model_util = ModelUtil(copy.deepcopy(self.__original_model))

        if model_util.has_sub_module(torch.quantization.QuantStub):
            quant_model = model_util.model
        else:
            quant_model = torch.quantization.QuantWrapper(model_util.model)
        quant_model.cpu()
        quant_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

        if hasattr(quant_model, "fuse_model"):
            quant_model.fuse_model()
        else:
            torch.quantization.fuse_modules(
                quant_model,
                QuantizationAwareTraining.get_fused_modules(quant_model),
                inplace=True,
            )
        torch.quantization.prepare_qat(quant_model, inplace=True)
        get_logger().debug("quant_model is %s", quant_model)
        trainer.set_model(quant_model)

    @property
    def quantized_model(self) -> torch.nn.Module:
        return self.__quantized_model

    def get_quantized_parameters(self) -> dict:
        quantized_model = self.quantized_model
        get_logger().debug("quantized_model is %s", quantized_model)
        processed_modules = set()
        state_dict = quantized_model.state_dict()
        quantized_model_util = ModelUtil(quantized_model)
        parameter_dict: dict = dict()
        for k in state_dict:
            get_logger().debug("attribute is %s", k)
            if k in ("scale", "zero_point", "quant.scale", "quant.zero_point"):
                continue
            if "." not in k:
                get_logger().debug("skip attribute is %s", k)
                continue
            module_name = ".".join(k.split(".")[:-1])
            if module_name in processed_modules:
                continue
            if not quantized_model_util.has_attr(module_name):
                continue
            sub_module = quantized_model_util.get_attr(module_name)
            if module_name.startswith("module."):
                module_name = module_name[len("module."):]
            if isinstance(
                sub_module,
                (
                    torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d,
                    torch.nn.quantized.modules.linear.Linear,
                    torch.nn.quantized.modules.conv.Conv2d,
                ),
            ):
                weight, bias = sub_module._weight_bias()
                assert weight.is_quantized
                scale = weight.q_per_channel_scales()
                zero_point = weight.q_per_channel_zero_points()
                weight = weight.detach().int_repr()
                parameter_dict[module_name + ".weight"] = (weight, scale, zero_point)
                if bias is not None:
                    bias = bias.detach()
                    parameter_dict[module_name + ".bias"] = bias
                processed_modules.add(module_name)
                continue
            if isinstance(
                sub_module,
                (torch.nn.quantized.modules.batchnorm.BatchNorm2d),
            ):
                weight = sub_module.weight.detach()
                assert not weight.is_quantized
                bias = sub_module.bias.detach()
                assert not bias.is_quantized
                running_mean = sub_module.running_mean.detach()
                assert not running_mean.is_quantized
                running_var = sub_module.running_var.detach()
                assert not running_var.is_quantized

                parameter_dict[module_name + ".weight"] = weight
                parameter_dict[module_name + ".bias"] = bias
                parameter_dict[module_name + ".running_mean"] = running_mean
                parameter_dict[module_name + ".running_var"] = running_var
                processed_modules.add(module_name)
                continue
            if not isinstance(
                sub_module, torch.nn.quantized.modules.linear.LinearPackedParams
            ):
                get_logger().warning("unsupported sub_module type %s", type(sub_module))

        return parameter_dict

    def load_quantized_parameters(self, parameter_dict: dict) -> dict:
        model_util = ModelUtil(self.__original_model)
        processed_modules = set()
        state_dict = self.__quantized_model.state_dict()
        quantized_model_util = ModelUtil(self.__quantized_model)
        for name, module in self.__original_model.named_modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                get_logger().debug("ignore BatchNorm2d %s", name)
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.zeros_(module.running_mean)
                torch.nn.init.ones_(module.running_var)
                # module.eps = 0

        for k in state_dict:
            if k in ("scale", "zero_point", "quant.scale", "quant.zero_point"):
                continue
            if "." not in k:
                continue
            module_name = ".".join(k.split(".")[:-1])
            if module_name in processed_modules:
                continue
            if not quantized_model_util.has_attr(module_name):
                continue
            sub_module = quantized_model_util.get_attr(module_name)
            if module_name.startswith("module."):
                module_name = module_name[len("module."):]
            if isinstance(
                sub_module,
                (
                    torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d,
                    torch.nn.quantized.modules.linear.Linear,
                    torch.nn.quantized.modules.conv.Conv2d,
                    torch.nn.quantized.modules.batchnorm.BatchNorm2d,
                ),
            ):
                processed_modules.add(module_name)
                weight = parameter_dict[module_name + ".weight"]
                if isinstance(weight, tuple):
                    (weight, scale, zero_point) = weight
                    weight = weight.float()
                    for idx, v in enumerate(weight):
                        weight[idx] = (v - zero_point[idx]) * scale[idx]
                model_util.set_attr(module_name + ".weight", weight)

                for suffix in [".bias", ".running_mean", ".running_var"]:
                    attr_name = module_name + suffix
                    if attr_name in parameter_dict:
                        model_util.set_attr(attr_name, parameter_dict[attr_name])
                continue
            if not isinstance(
                sub_module, torch.nn.quantized.modules.linear.LinearPackedParams
            ):
                get_logger().warning("unsupported sub_module type %s", type(sub_module))
        return parameter_dict

    @staticmethod
    def get_fused_modules(model):
        modules = list(model.named_modules())
        list_of_list = []
        i = 0
        while i < len(modules):
            candidates: set = set(DEFAULT_OP_LIST_TO_FUSER_METHOD.keys())
            j = i
            end_index = None
            while j < len(modules):
                module = modules[j][1]
                new_candidates = set()
                for candidate in candidates:
                    if isinstance(module, candidate[0]):
                        if len(candidate) == 1:
                            end_index = j
                        else:
                            new_candidates.add(candidate[1:])
                if not new_candidates:
                    break
                candidates = new_candidates
                j += 1
            if end_index is not None:
                module_name_list = []
                while i <= end_index:
                    module_name_list.append(modules[i][0])
                    i += 1
                list_of_list.append(module_name_list)
            else:
                i += 1
        get_logger().debug("list_of_list is %s", list_of_list)
        return list_of_list
