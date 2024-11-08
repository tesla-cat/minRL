# cspell:ignore logp
from typing import Dict, List, Union

import numpy as np
import torch as tc
import torch.nn as nn
from torch.distributions import Categorical, Normal

NETS_TYPE = Dict[str, Dict[str, Union[nn.Module, List[float]]]]
PI_DIC_TYPE = Dict[str, Union[Categorical, Normal]]
A_DIC_TYPE = Dict[str, np.ndarray]


class MixedActor(nn.Module):
    def __init__(s, nets: NETS_TYPE):
        super(MixedActor, s).__init__()
        s.nets = nets
        for k, x in nets.items():
            s.add_module(k, x["net"])

    def forward(s, o) -> Dict[str, tc.Tensor]:
        v = s.nets["shared"]["net"](o)
        return {k: x["net"](v) for k, x in s.nets.items() if k != "shared"}

    def get_pi(s, o):
        pi_dic: PI_DIC_TYPE = {}
        for k, logits in s.forward(o).items():
            if "values" in s.nets[k]:
                pi = Categorical(logits=logits)
            else:
                mean, log_std = logits.T
                pi = Normal(mean, log_std.exp())
            pi_dic[k] = pi
        return pi_dic

    def get_logp(s, pi_dic: PI_DIC_TYPE, a_dic) -> tc.Tensor:
        return sum([pi.log_prob(a_dic[k]) for k, pi in pi_dic.items()])

    def map_actions(s, a_dic: A_DIC_TYPE):
        a2_dic: A_DIC_TYPE = {}
        for k, a in a_dic.items():
            if "values" in s.nets[k]:
                values = s.nets[k]["values"]
                a2 = np.array([values[i] for i in a])
            else:
                L, H = s.nets[k]["range"]
                a2 = L + (np.tanh(a) + 1) / 2 * (H - L)
            a2_dic[k] = a2
        return a2_dic


if __name__ == "__main__":
    from minRL.utils.nn_utils import mlp

    nets = {
        "shared": {"net": mlp([3, 64, 64])},
        "discrete_1": {
            "values": [1, 2, 3],
            "net": mlp([64, 3]),
        },
        "continuous_2": {
            "range": [0, 1],
            "net": mlp([64, 2]),
        },
    }

    pi_net = MixedActor(nets)
