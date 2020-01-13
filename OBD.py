#!/usr/bin/env python3

from .gradient import get_gradient


def optimal_brain_damage(model, loss):
    gradient = get_gradient(model, loss)
    print(gradient.sort())
