import torch
import torch.nn as nn


class CrossEntropyLossWithCoefficents(nn.CrossEntropyLoss):
    def __init__(self, coefficent_init_value=1, reduction="mean"):
        super(
            CrossEntropyLossWithCoefficents,
            self).__init__(
            reduction="none",
        )
        self.coefficent_init_value = coefficent_init_value
        self.final_reduction = reduction

    def forward(self, input, target):
        loss = super().forward(input, target)
        loss_cnt = loss.shape[0]
        coefficents = []
        for _ in range(loss_cnt):
            coefficent = torch.Tensor(
                [self.coefficent_init_value], requires_grad=True)
            coefficents.append(coefficent)
        if self.final_reduction == "none":
            return torch.mul(coefficent, loss)
        result = loss @ coefficents
        if self.final_reduction == "mean":
            result /= loss_cnt
        return result
