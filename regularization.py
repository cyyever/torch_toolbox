#!/usr/bin/env python3


def add_l2_regularization(loss_fun, model, weight_decay):
    def loss_fun_with_regularization(*args, **kwargs):
        loss = loss_fun(*args, **kwargs)
        for param in model.parameters():
            loss += (param ** 2).sum()
        loss *= weight_decay
        return loss

    return loss_fun_with_regularization
