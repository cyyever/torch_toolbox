import copy
import functools
from typing import Callable

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.reflection import get_kwarg_names
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from cyy_torch_toolbox.dependency import has_hugging_face, has_torch_geometric

if has_torch_geometric:
    import torch_geometric.nn
    import torch_geometric.utils

if has_hugging_face:
    import transformers

from cyy_torch_toolbox.dataset_util import GraphDatasetUtil
from cyy_torch_toolbox.device import get_devices
from cyy_torch_toolbox.ml_type import MachineLearningPhase, ModelType
# from cyy_torch_toolbox.model_transform.checkpointed_model import \
#     get_checkpointed_model
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.tensor import tensor_to


class ModelWithLoss:
    """
    Combine a model with a loss function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: None | ModelType = None,
        loss_fun: str | Callable | None = None,
    ):
        self._original_model: torch.nn.Module = model
        self._model: torch.nn.Module = model
        self.__loss_fun: Callable | None = None
        if loss_fun is not None:
            self.set_loss_fun(loss_fun)
        self.__model_type = model_type
        self.need_input_features = False
        self.need_cpu_inputs = False

    def compile_model(self) -> None:
        raise NotImplementedError()
        # self._model = torch.compile(self._original_model)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @functools.cached_property
    def model_util(self) -> ModelUtil:
        return ModelUtil(self.model)

    @property
    def model_type(self) -> ModelType | None:
        return self.__model_type

    @property
    def loss_fun(self) -> Callable:
        if self.__loss_fun is None:
            self.__loss_fun = self.__choose_loss_function()
        return self.__loss_fun

    def set_loss_fun(self, loss_fun: Callable | str) -> None:
        if isinstance(loss_fun, str):
            if loss_fun == "CrossEntropyLoss":
                self.__loss_fun = nn.CrossEntropyLoss()
            else:
                raise RuntimeError(f"unknown loss function {loss_fun}")
        else:
            self.__loss_fun = loss_fun

    def offload_from_gpu(self):
        self.model.zero_grad(set_to_none=True)
        self.to(device="cpu")

    def get_input_feature(self, inputs):
        if hasattr(self.model, "get_input_feature"):
            return self.model.get_input_feature(inputs)
        return None

    def __call__(
        self,
        inputs,
        targets,
        phase: MachineLearningPhase | None = None,
        device=None,
        non_blocking: bool = False,
        input_features=None,
        need_backward: bool = False,
        **kwargs,
    ) -> dict:
        if phase is not None:
            self.__set_model_mode(
                is_training=(phase == MachineLearningPhase.Training),
                need_backward=need_backward,
            )

        # DALI returns nested targets
        if len(targets.shape) > 1:
            targets = targets.view(-1).long()

        cpu_inputs = None
        if device is not None:
            if self.need_cpu_inputs:
                cpu_inputs = tensor_to(inputs, device="cpu", non_blocking=non_blocking)
            if input_features is not None:
                input_features = tensor_to(
                    input_features, device=device, non_blocking=non_blocking
                )
            else:
                inputs = tensor_to(inputs, device=device, non_blocking=non_blocking)
            targets = tensor_to(targets, device=device, non_blocking=non_blocking)
            self.to(device=device, non_blocking=non_blocking)

        if input_features is None and self.need_input_features:
            input_features = self.get_input_feature(inputs)
            assert input_features is not None
        output = self._forward_model(
            inputs=inputs, input_features=input_features, targets=targets, **kwargs
        )

        match output:
            case torch.Tensor():
                match self.loss_fun:
                    case nn.BCEWithLogitsLoss():
                        output = output.view(-1)
                        targets = targets.to(
                            dtype=output.dtype, non_blocking=non_blocking
                        )
                loss = self.loss_fun(output, targets)
                output = {"loss": loss, "classification_output": output}
        is_averaged_loss = self.__is_averaged_loss()
        if is_averaged_loss is None:
            is_averaged_loss = output["classification_output"] is not None
        return output | {
            "inputs": inputs,
            "cpu_inputs": cpu_inputs,
            "input_features": input_features,
            "targets": targets,
            "is_averaged_loss": is_averaged_loss,
        }

    def _forward_model(
        self, inputs, input_features=None, **kwargs
    ) -> dict | torch.Tensor:
        fun: Callable = self.model
        if hasattr(self.model, "forward_input_feature") and input_features is not None:
            fun = self.model.forward_input_feature
            real_inputs = input_features
        else:
            real_inputs = inputs
        match real_inputs:
            case torch.Tensor():
                return fun(real_inputs)
            case tuple():
                return fun(*real_inputs)
            case dict():
                return fun(**real_inputs)
            case _:
                raise NotImplementedError(type(real_inputs))

    def replace_model(self, model):
        return ModelWithLoss(
            model=model,
            loss_fun=self.loss_fun,
            model_type=self.model_type,
        )

    def to(self, device, non_blocking=False):
        for param in self.model.parameters():
            if param.device != device:
                self.model.to(device=device, non_blocking=non_blocking)
            break
        for buffer in self.model.buffers():
            if buffer.device != device:
                self.model.to(device=device, non_blocking=non_blocking)
            return

    def __choose_loss_function(self) -> torch.nn.modules.loss._Loss:
        layers = [
            m
            for _, m in self.model_util.get_modules()
            if not isinstance(
                m,
                (
                    torch.quantization.QuantStub,
                    torch.quantization.DeQuantStub,
                    torch.quantization.stubs.QuantWrapper,
                    torch.quantization.fake_quantize.FakeQuantize,
                    torch.quantization.observer.MovingAverageMinMaxObserver,
                    torch.quantization.observer.MovingAveragePerChannelMinMaxObserver,
                    torch.nn.modules.dropout.Dropout,
                ),
            )
            and "MemoryEfficientSwish" not in str(m)
        ]
        last_layer = layers[-1]

        get_logger().debug("last module is %s", last_layer.__class__)
        if isinstance(last_layer, nn.LogSoftmax):
            get_logger().debug("choose loss function NLLLoss")
            return nn.NLLLoss()
        if isinstance(last_layer, nn.Linear):
            if last_layer.out_features == 1:
                get_logger().debug("choose loss function BCEWithLogitsLoss")
                return nn.BCEWithLogitsLoss()
            get_logger().debug("choose loss function CrossEntropyLoss")
            return nn.CrossEntropyLoss()
        get_logger().error("can't choose a loss function, model is %s", self._model)
        raise NotImplementedError(type(last_layer))

    def get_underlying_model(self):
        model = self._original_model
        if isinstance(model, torch.quantization.stubs.QuantWrapper):
            return model.module
        return model

    def __is_averaged_loss(self) -> bool | None:
        if hasattr(self.loss_fun, "reduction"):
            if self.loss_fun.reduction in ("mean",):
                return True
            return False
        return None

    def __repr__(self):
        return f"model: {self._model.__class__.__name__}, loss_fun: {self.loss_fun}"

    def __set_model_mode(self, is_training: bool, need_backward: bool = False) -> None:
        if is_training:
            if self._model.training:
                return
            self._model.train()
            return
        if self._model.training:
            self._model.eval()
            if need_backward:
                self.model_util.change_modules(
                    f=lambda _, module, __: module.train(), module_type=nn.RNNBase
                )


# class CheckPointedModelWithLoss:
#     def __init__(self, model_with_loss: ModelWithLoss):
#         self.__model_with_loss = model_with_loss.replace_model(
#             get_checkpointed_model(model_with_loss.model)
#         )

#     def __getattr__(self, attr):
#         return getattr(self.__model_with_loss, attr)

#     def __call__(self, **kwargs) -> dict:
#         phase = kwargs["phase"]
#         if phase == MachineLearningPhase.Training:
#             input_features = kwargs.get("input_features", None)
#             if input_features is not None:
#                 input_features.requires_grad_()
#             inputs = kwargs.get("inputs", None)
#             if inputs is not None:
#                 inputs.requires_grad_()
#         return self.__model_with_loss.__call__(**kwargs)


class VisionModelWithLoss(ModelWithLoss):
    pass
    # def __call__(self, inputs, **kwargs):
    #     inputs = tensor_to(inputs, non_blocking=True, memory_format=torch.channels_last)
    #     return super().__call__(inputs=inputs, **kwargs)

    # def to(self, device, non_blocking=False):
    #     self.model.to(non_blocking=non_blocking, memory_format=torch.channels_last)
    #     super().to(device=device, non_blocking=non_blocking)


class TextModelWithLoss(ModelWithLoss):
    @property
    def __is_hugging_face_model(self) -> bool:
        return has_hugging_face and isinstance(
            self.model, transformers.modeling_utils.PreTrainedModel
        )

    def get_input_feature(self, inputs):
        res = super().get_input_feature(inputs)
        if res is not None:
            return res
        if self.__is_hugging_face_model:
            match inputs:
                case transformers.tokenization_utils_base.BatchEncoding() | dict():
                    input_ids = inputs["input_ids"]
                case _:
                    input_ids = inputs
            if hasattr(self.model, "distilbert"):
                if len(list(input_ids.shape)) == 1:
                    input_ids = input_ids.unsqueeze(dim=0)
                return self.model.distilbert.embeddings(input_ids).detach()
            if hasattr(self.model, "bert"):
                return self.model.get_input_embeddings()(input_ids).detach()
            raise NotImplementedError(self.model)
        return None

    def _forward_model(
        self, inputs, targets, input_features=None, **kwargs
    ) -> dict | torch.Tensor:

        if hasattr(self.model, "forward"):
            kwarg_names = get_kwarg_names(self.model.forward)
        else:
            kwarg_names = get_kwarg_names(self.model)
        if "input_ids" in kwarg_names and "inputs_embeds" in kwarg_names:
            if input_features is not None:
                if inputs is not None:
                    new_inputs = copy.copy(inputs)
                    new_inputs.pop("input_ids", None)
                else:
                    new_inputs = {}
                new_inputs["inputs_embeds"] = input_features
                output = self.model(**new_inputs, labels=targets)
            else:
                output = self.model(**inputs, labels=targets)
            return {
                "loss": output["loss"],
                "classification_output": output["logits"],
            }
        return super()._forward_model(
            inputs=inputs, targets=targets, input_features=input_features, **kwargs
        )


class GraphModelWithLoss(ModelWithLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_and_neighbour_index_map = {}
        self.node_index_map = {}
        self.edge_index_map = {}
        self.neighbour_hop = sum(
            [
                1
                for _, module in self.model_util.get_modules()
                if isinstance(module, torch_geometric.nn.MessagePassing)
            ]
        )

    def __call__(self, **kwargs) -> dict:
        inputs = kwargs["inputs"]
        inputs["x"] = inputs["x"][0]
        inputs["edge_index"] = inputs["edge_index"][0]
        mask = kwargs.pop("mask", None)
        if mask is None:
            return super().__call__(**kwargs)
        phase = kwargs["phase"]
        mask = mask.view(-1)
        if phase not in self.node_index_map:
            edge_index = inputs["edge_index"]
            node_indices = set(torch_geometric.utils.mask_to_index(mask).tolist())
            self.node_index_map[phase] = node_indices
            node_and_neighbours = GraphDatasetUtil.get_neighbors_from_edges(
                node_indices=node_indices, edge_index=edge_index, hop=self.neighbour_hop
            )
            tmp = set(node_and_neighbours.keys())
            for value in node_and_neighbours.values():
                tmp |= value
            node_and_neighbours = tmp
            node_and_neighbour_index_map = {
                node_index: idx
                for idx, node_index in enumerate(sorted(node_and_neighbours))
            }
            self.node_and_neighbour_index_map[phase] = node_and_neighbour_index_map
            new_source_list = []
            new_target_list = []
            for source, target in GraphDatasetUtil.foreach_edge(edge_index):
                if source in node_indices or target in node_indices:
                    new_source_list.append(node_and_neighbour_index_map[source])
                    new_target_list.append(node_and_neighbour_index_map[target])
            edge_index = torch.tensor(
                data=[new_source_list, new_target_list], dtype=edge_index.dtype
            )
            self.edge_index_map[phase] = edge_index
        else:
            node_indices = self.node_index_map[phase]
            node_and_neighbour_index_map = self.node_and_neighbour_index_map[phase]
            node_and_neighbours = set(node_and_neighbour_index_map.keys())
            edge_index = self.edge_index_map[phase]
        inputs["edge_index"] = edge_index
        kwargs["targets"] = kwargs["targets"][mask]
        new_mask = torch.zeros_like(mask)
        print("node num,", len(node_indices), len(node_and_neighbours))
        for idx in node_and_neighbours:
            new_mask[idx] = True
        print("before x shape", inputs["x"].shape)
        inputs["x"] = inputs["x"][new_mask]
        print("after x shape", inputs["x"].shape)
        new_mask = torch.zeros((len(node_and_neighbours),), dtype=torch.bool)
        for idx in node_indices:
            new_mask[node_and_neighbour_index_map[idx]] = True
        kwargs["mask"] = new_mask
        return super().__call__(**kwargs)

    def _forward_model(self, mask, **kwargs) -> dict | torch.Tensor:
        output = super()._forward_model(**kwargs)
        return output[mask]


class ParallelModelWithLoss(ModelWithLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert torch.cuda.is_available()
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="tcp://127.0.0.1:23456",
                rank=0,
                world_size=len(get_devices()),
            )
        self._original_model = self._model
        self._model = DDP(self._original_model)

    @classmethod
    def create(cls, model_with_loss: ModelWithLoss):
        return cls(
            model=model_with_loss.model,
            loss_fun=model_with_loss.loss_fun,
            model_type=model_with_loss.model_type,
        )
