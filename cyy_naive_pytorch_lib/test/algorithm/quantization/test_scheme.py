#!/usr/bin/env python3
import torch

from algorithm.quantization.scheme import stochastic_quantization


def test_stochastic_quantization():
    a = torch.rand(2, 2)
    res = stochastic_quantization(256)
    pair = res[0](a)
    recovered_tensor = res[1](pair)
    print(
        "recovered_tensor",
        recovered_tensor,
        "tensor",
        a,
        "norm",
        torch.linalg.norm(recovered_tensor - a),
    )
    res = stochastic_quantization(256, use_l2_norm=True)
    pair = res[0](a)
    recovered_tensor = res[1](pair)
    print(
        "use_l2_norm recovered_tensor",
        recovered_tensor,
        "tensor",
        a,
        "norm",
        torch.linalg.norm(recovered_tensor - a),
    )
