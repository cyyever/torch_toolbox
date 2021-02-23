from typing import Callable, Tuple

import torch


def stochastic_quantization(quantization_level: int) -> Tuple[Callable, Callable]:
    """Implement Stochastic Quantization as described in QSGD: Communication-Efficient SGDvia Gradient Quantization and Encoding (https://arxiv.org/pdf/1610.02132.pdf)"""

    def quant(tensor: torch.Tensor):
        nonlocal quantization_level
        tensor = tensor.reshape(-1)
        assert len(tensor.shape) == 1
        l2_norm = torch.norm(tensor)
        assert l2_norm != 0
        sign_tensor = torch.sign(tensor)
        normalized_tensor = tensor.abs() / l2_norm
        for idx, element in enumerate(normalized_tensor):
            assert element <= 1
            for slot in range(quantization_level):
                if element <= ((slot + 1) / quantization_level):
                    prob = element * quantization_level / l2_norm - slot
                    assert prob >= 0 and prob <= 1
                    m = torch.distributions.Bernoulli(prob)
                    if m.sample() == 1:
                        normalized_tensor[idx] = (slot + 1) / quantization_level
                    else:
                        normalized_tensor[idx] = slot / quantization_level

        return (l2_norm, sign_tensor, normalized_tensor)

    def dequant():
        return 1

    return (quant(), dequant())
