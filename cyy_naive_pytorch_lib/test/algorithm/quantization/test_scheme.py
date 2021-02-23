#!/usr/bin/env python3
import torch

from algorithm.quantization.scheme import stochastic_quantization


def test_stochastic_quantization():
    a = torch.rand(3, 5)
    res = stochastic_quantization(256)
    res[0](a)
