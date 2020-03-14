import torch
import torch.nn as nn


class CrossEntropyLossWithCoefficents(nn.CrossEntropyLoss):
    def __init__(self, coefficent_init_value=1, final_reduction="mean"):
        super(
            CrossEntropyLossWithCoefficents,
            self).__init__(
            reduction="none",
        )
        self.__coefficent_init_value = coefficent_init_value
        self.__final_reduction = final_reduction
        self.device = None
        self.__coefficients = []

    def get_coefficients(self):
        return self.__coefficients

    def forward(self, input, target):
        self.reduction = "none"
        loss = super().forward(input, target)
        loss_cnt = loss.shape[0]
        self.__coefficients = []
        for _ in range(loss_cnt):
            coefficient = torch.Tensor([self.__coefficent_init_value])
            coefficient.requires_grad_(True)
            if self.device is not None:
                coefficient = coefficient.to(self.device)
            self.__coefficients.append(coefficient)
        if self.__final_reduction == "none":
            return torch.mul(self.__coefficients, loss)
        result = loss @ torch.stack(self.__coefficients)
        if self.__final_reduction == "mean":
            result /= loss_cnt
        self.reduction = self.__final_reduction
        return result
