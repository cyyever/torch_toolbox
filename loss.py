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

    def forward(self, input, target):
        self.reduction = "none"
        loss = super().forward(input, target)
        loss_cnt = loss.shape[0]
        coefficents = []
        for _ in range(loss_cnt):
            coefficent = torch.Tensor([self.__coefficent_init_value])
            coefficent.requires_grad_(True)
            if self.device is not None:
                coefficent = coefficent.to(self.device)
            coefficents.append(coefficent)
        if self.__final_reduction == "none":
            return torch.mul(coefficent, loss)
        result = loss @ torch.stack(coefficents)
        if self.__final_reduction == "mean":
            result /= loss_cnt
        self.reduction = self.__final_reduction
        return result
