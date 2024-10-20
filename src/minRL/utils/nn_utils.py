import numpy as np
import torch as tc
import torch.nn as nn


def discount_cum_sum(x: np.ndarray, g):
    """
    verified
    in: x0, x1, x2
    out: x0 + g(x1 + g x2), x1 + g x2, x2
    """
    out = x.copy()
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + g * out[i + 1]
    return out


def tensor(x):
    return tc.as_tensor(x, dtype=tc.float32)


def mlp(sizes, act=nn.Tanh, act_out=nn.Identity):
    # verified
    layers = [[nn.Linear(sizes[i], sizes[i + 1]), act()] for i in range(len(sizes) - 1)]
    layers = sum(layers, start=[])
    layers[-1] = act_out()
    return nn.Sequential(*layers)
