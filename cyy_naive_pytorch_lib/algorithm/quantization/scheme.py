from typing import Callable, Tuple

import torch


def stochastic_quantization(
    quantization_level: int, use_l2_norm: bool = False
) -> Tuple[Callable, Callable]:
    """Implement Stochastic Quantization as described in QSGD: Communication-Efficient SGDvia Gradient Quantization and Encoding (https://arxiv.org/pdf/1610.02132.pdf)"""

    def quant(tensor: torch.Tensor):
        nonlocal quantization_level, use_l2_norm
        tensor_shape = tensor.shape
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
        for idx, element in enumerate(normalized_abs_tensor):
            assert element <= 1
            for slot in range(quantization_level):
                if (
                    (slot / quantization_level)
                    <= element
                    <= ((slot + 1) / quantization_level)
                ):
                    prob = element * quantization_level - slot
                    assert 0 <= prob <= 1
                    m = torch.distributions.Bernoulli(prob)
                    if m.sample() == 1:
                        normalized_abs_tensor[idx] = slot + 1
                    else:
                        normalized_abs_tensor[idx] = slot
                    break

        return (
            norm,
            sign_tensor,
            normalized_abs_tensor.int(),
            quantization_level,
            tensor_shape,
        )

    def dequant(quantized_pair):
        (
            norm,
            sign_tensor,
            quantized_tensor,
            quantization_level,
            tensor_shape,
        ) = quantized_pair
        quantized_tensor = quantized_tensor.float()
        quantized_tensor *= norm
        res = quantized_tensor * sign_tensor / quantization_level
        res.reshape(tensor_shape)
        return res

    return (quant, dequant)
