from typing import Callable, Tuple

import cyy_naive_cpp_extension
import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order

from device import put_data_to_device
from tensor import cat_tensors_to_vector, split_tensor_to_dict


def stochastic_quantization(
    quantization_level: int, use_l2_norm: bool = False
) -> Tuple[Callable, Callable]:
    """Implement Stochastic Quantization as described in QSGD: Communication-Efficient SGDvia Gradient Quantization and Encoding (https://arxiv.org/pdf/1610.02132.pdf)"""

    def quant(tensor):
        name_and_shapes = None
        if isinstance(tensor):
            name_and_shapes = []
            for k in sorted(tensor.keys()):
                name_and_shapes.append((k, tensor[k].shape))
            tensor = cat_tensors_to_vector(get_mapping_values_by_key_order(tensor))

        old_tensor_shape = tensor.shape
        old_device = tensor.device
        tensor = put_data_to_device(tensor)
        tensor = tensor.reshape(-1)
        assert len(tensor.shape) == 1

        norm = None
        if use_l2_norm:
            norm = torch.linalg.norm(tensor)
        else:
            norm = torch.linalg.norm(tensor, ord=float("inf"))
        assert norm > 0
        sign_tensor = torch.sign(tensor)
        normalized_abs_tensor = tensor.abs() / norm
        slot_tensor = cyy_naive_cpp_extension.torch.stochastic_quantization(
            normalized_abs_tensor, quantization_level
        )
        prob_tensor = normalized_abs_tensor * quantization_level - slot_tensor
        random_vector = torch.distributions.Bernoulli(prob_tensor).sample()
        slot_tensor += random_vector

        norm = put_data_to_device(norm, old_device)
        sign_tensor = put_data_to_device(sign_tensor.char(), old_device).reshape(
            old_tensor_shape
        )
        slot_tensor = put_data_to_device(slot_tensor.byte(), old_device).reshape(
            old_tensor_shape
        )

        return (norm, sign_tensor, slot_tensor, quantization_level, name_and_shapes)

    def dequant(quantized_pair):
        (
            norm,
            sign_tensor,
            quantized_tensor,
            quantization_level,
            name_and_shapes,
        ) = quantized_pair

        quantized_tensor = quantized_tensor.float()
        quantized_tensor *= norm
        res = quantized_tensor * sign_tensor / quantization_level

        if name_and_shapes is not None:
            res = split_tensor_to_dict(name_and_shapes, res)
        return res

    return (quant, dequant)
