#!/usr/bin/env python3
import torch

from algorithm.quantization.scheme import stochastic_quantization


def test_stochastic_quantization():
    a = torch.rand(2, 100000)
    quant, dequant = stochastic_quantization(256)
    pair = quant(a)
    print(pair)
    recovered_tensor = dequant(pair)
    print(
        "recovered_tensor",
        recovered_tensor,
        "tensor",
        a,
        "relative diff",
        torch.linalg.norm(recovered_tensor - a) / torch.linalg.norm(a),
    )
    quant, dequant = stochastic_quantization(256, use_l2_norm=True)
    pair = quant(a)
    recovered_tensor = dequant(pair)
    print(
        "use_l2_norm recovered_tensor",
        recovered_tensor,
        "tensor",
        a,
        "relative diff",
        torch.linalg.norm(recovered_tensor - a) / torch.linalg.norm(a),
    )
