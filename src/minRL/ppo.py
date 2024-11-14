# cspell:ignore logp
from typing import Dict

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from torch.distributions import Categorical, Normal
from torch.optim import Adam

from minRL.utils import Recorder


def mlp(sizes, act=nn.Tanh, out=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    layers = layers[:-1] + [out()] if out else layers[:-1]
    return nn.Sequential(*layers)


def discount_cumsum(x: np.ndarray, g):
    x2 = x.copy()
    for i in reversed(range(len(x) - 1)):
        x2[i] += x2[i + 1] * g
    return x2


def tensor(x):
    if isinstance(x, dict):
        return {k: tensor(v) for k, v in x.items()}
    return tc.as_tensor(x, dtype=tc.float32)


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def add_record(D, N, i, dic: Dict):
    for k, v in dic.items():
        if k not in D:
            shape = v.shape if isinstance(v, np.ndarray) else ()
            D[k] = np.zeros((N, *shape), np.float32)
        D[k][i] = v


def get_envs():
    return list(gym.envs.registry.keys())


NETS_TYPE = Dict[str, Dict[str, nn.Module]]
PI_DIC_TYPE = Dict[str, Normal]


class MixedActor(nn.Module):
    def __init__(s, nets: NETS_TYPE):
        super().__init__()
        s.nets = nets
        s.log_std = {}

        for k, x in nets.items():
            s.add_module(k, x["net"])
            if "range" in x:
                s.log_std[k] = nn.Parameter(
                    tensor(-0.5 * np.ones(x["dim"], np.float32))
                )
                s.register_parameter(f"{k}_log_std", s.log_std[k])

    def get_pi(s, o):
        pi_dic: PI_DIC_TYPE = {}
        o2 = s.nets["shared"]["net"](o)
        for k, x in s.nets.items():
            if k != "shared":
                logits = x["net"](o2)
                if "values" in x:
                    pi_dic[k] = Categorical(logits=logits)
                else:
                    pi_dic[k] = Normal(logits, tc.exp(s.log_std[k]))
        return pi_dic

    def get_logp(s, pi_dic: PI_DIC_TYPE, a_dic):
        logp = 0
        for k, pi in pi_dic.items():
            lp: tc.Tensor = pi.log_prob(a_dic[k])
            logp += lp.sum(-1) if isinstance(pi, Normal) else lp
        return logp

    def map_actions(s, a_dic: Dict[str, np.ndarray]):
        a2_dic = {}
        for k, a in a_dic.items():
            if "values" in s.nets[k]:
                vals = s.nets[k]["values"]
                a2_dic[k] = np.array([vals[i] for i in a]) if a.shape else vals[a]
            else:
                L, H = s.nets[k]["range"]
                a2_dic[k] = L + (H - L) * (np.tanh(a) + 1) / 2
        return a2_dic


def make_actor_critic(env: gym.Env, hidden=[64, 64]):
    o_dim = env.observation_space.shape[0]
    a = env.action_space
    if isinstance(a, Box):
        a_dim = a.shape[0]
        act = {"range": [a.low, a.high], "dim": a_dim}
    elif isinstance(a, Discrete):
        a_dim = a.n
        act = {"values": range(a_dim)}
    act["net"] = mlp([o_dim, *hidden, a_dim])
    nets = {"shared": {"net": nn.Identity()}, "action": act}
    return MixedActor(nets), mlp([o_dim, *hidden, 1])


class PPOClip:
    def __init__(
        s,
        pi_net: MixedActor,
        V_net: nn.Module,
        pi_lr=3e-4,
        V_lr=1e-3,
        gam=0.99,
        lam=0.97,
    ):
        s.pi_net, s.V_net = pi_net, V_net
        s.pi_opt = Adam(pi_net.parameters(), lr=pi_lr)
        s.V_opt = Adam(V_net.parameters(), lr=V_lr)
        s.gam, s.lam = gam, lam

    def get_V(s, o):
        return tc.squeeze(s.V_net(o), -1)

    def get_a_logp_V(s, o):
        with tc.no_grad():
            o = tensor(o)
            pi_dic = s.pi_net.get_pi(o)
            a_dic = {k: pi.sample() for k, pi in pi_dic.items()}
            logp = s.pi_net.get_logp(pi_dic, a_dic)
            a_dic = {k: a.numpy() for k, a in a_dic.items()}
            return a_dic, logp.numpy(), s.get_V(o).numpy()

    def get_D_from_env(s, env: gym.Env, rec: Recorder, N=50, max_L=50):
        D = {}
        o, _ = env.reset(seed=0)
        R, L = 0, 0
        for i in range(N):
            a_dic, logp, V = s.get_a_logp_V(o)
            a2_dic = s.pi_net.map_actions(a_dic)
            a = a2_dic["action"] if "action" in a2_dic else a2_dic
            o_next, r, done, truncated, _ = env.step(a)
            R, L = R + r, L + 1
            V_next = 0
            ended = done or L == max_L or i == N - 1
            if ended:
                V_next = 0 if done else s.get_a_logp_V(o_next)[-1]
                o_next, _ = env.reset(seed=0)
                if done or L == max_L:
                    id = env.spec.id if isinstance(env, gym.Env) else env.id
                    rec.add(f"my_ppo|{id}", [R, L])
                R, L = 0, 0
            dic = dict(o=o, r=r, ended=ended, logp=logp, V=V, V_next=V_next)
            add_record(D, N, i, {**dic, **a_dic})
            o = o_next
        D["a_dic"] = {k: D[k] for k in a_dic}
        return D

    def find_R_A(s, D):
        start, N = 0, len(D["r"])
        for k in ["R", "A"]:
            D[k] = np.zeros(N, np.float32)
        for i in range(N):
            if D["ended"][i]:
                V_next = D["V_next"][i]
                slc = slice(start, i + 1)
                r = np.append(D["r"][slc], V_next)
                V = np.append(D["V"][slc], V_next)
                A = r[:-1] + s.gam * V[1:] - V[:-1]
                D["A"][slc] = discount_cumsum(A, s.gam * s.lam)
                D["R"][slc] = discount_cumsum(r, s.gam)[:-1]
                start = i + 1
        D["A"] = normalize(D["A"])
        return tensor(D)

    def learn(s, D, steps=80, eps=0.2, kl_max=0.015):
        D = s.find_R_A(D)
        o, a_dic, A, lp_old, R = D["o"], D["a_dic"], D["A"], D["logp"], D["R"]
        for _ in range(steps):
            s.pi_opt.zero_grad()
            pi_dic = s.pi_net.get_pi(o)
            lp = s.pi_net.get_logp(pi_dic, a_dic)
            ratio = tc.exp(lp - lp_old)
            r_clip = tc.clamp(ratio, 1 - eps, 1 + eps)
            loss_pi = -tc.min(ratio * A, r_clip * A).mean()
            kl = (lp_old - lp).mean().item()
            if kl > kl_max:
                break
            loss_pi.backward()
            s.pi_opt.step()

        for _ in range(steps):
            s.V_opt.zero_grad()
            loss_V = ((s.get_V(o) - R) ** 2).mean()
            loss_V.backward()
            s.V_opt.step()
        return loss_pi, loss_V
