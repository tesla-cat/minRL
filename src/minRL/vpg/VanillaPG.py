# cspell:ignore logp
from typing import Dict, Union

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
from torch.optim import Adam

import minRL.utils.mpi_fake as mpi
from minRL.utils.nn_utils import add_record, discount_cum_sum, tensor
from minRL.vpg.MixedActor import MixedActor

BUFFER_TYPE = Dict[str, Union[np.ndarray, tc.Tensor]]


class VanillaPG:
    def __init__(
        s,
        pi_net: MixedActor,
        V_net: nn.Module,
        pi_lr=3e-4,
        V_lr=1e-3,
        V_iters=80,
        gam=0.99,
        lam=0.97,
    ):
        s.pi_net, s.V_net = pi_net, V_net
        s.V_iters = V_iters
        s.gam, s.lam = gam, lam

        s.pi_params = list(s.pi_net.parameters())
        s.V_params = list(s.V_net.parameters())
        s.pi_opt = Adam(s.pi_params, lr=pi_lr)
        s.V_opt = Adam(s.V_params, lr=V_lr)
        mpi.sync_params(s.pi_params + s.V_params)

    def train_once(s, D: BUFFER_TYPE):
        # requires: r, ended, V, V_next, o, a_dic, logp
        D = s.find_R_and_A(D)
        s.update_pi(D)
        s.update_V(D)

    def find_R_and_A(s, D: BUFFER_TYPE):
        # 4, 5: compute R and A
        for k in ["R", "A"]:
            D[k] = np.zeros_like(D["r"])
        start, N = 0, len(D["r"])
        for i in range(N):
            if D["ended"][i]:
                V_next = D["V_next"][i]
                slc = slice(start, i + 1)
                r = np.append(D["r"][slc], V_next)
                V = np.append(D["V"][slc], V_next)
                # GAE-lambda advantage estimation
                A = r[:-1] + s.gam * V[1:] - V[:-1]
                D["A"][slc] = discount_cum_sum(A, s.gam * s.lam)
                D["R"][slc] = discount_cum_sum(r, s.gam)[:-1]
                start = i + 1
        D["A"] = mpi.normalize(D["A"])
        return tensor(D)

    def update_pi(s, D: BUFFER_TYPE):
        # 6, 7: estimate pg and optimize
        o, a_dic, A = D["o"], D["a_dic"], D["A"]
        s.pi_opt.zero_grad()
        pi_dic = s.pi_net.get_pi(o)
        logp = s.pi_net.get_logp(pi_dic, a_dic)
        loss = -(logp * A).mean()
        loss.backward()
        mpi.avg_grads(s.pi_params)
        s.pi_opt.step()

    def update_V(s, D: BUFFER_TYPE):
        # 8: fit V
        o, R = D["o"], D["R"]
        for _ in range(s.V_iters):
            s.V_opt.zero_grad()
            loss = ((s.get_V(o) - R) ** 2).mean()
            loss.backward()
            mpi.avg_grads(s.V_params)
            s.V_opt.step()

    def get_V(s, o):
        return tc.squeeze(s.V_net(o), -1)

    def get_action(s, o):
        with tc.no_grad():
            o = tensor(o)
            pi_dic = s.pi_net.get_pi(o)
            a_dic = {k: pi.sample() for k, pi in pi_dic.items()}
            logp = s.pi_net.get_logp(pi_dic, a_dic)
            a_dic = {k: v.numpy() for k, v in a_dic.items()}
            return a_dic, logp.numpy(), s.get_V(o).numpy()

    # ===============================================

    def get_D_from_env(s, env: gym.Env, N=4000):
        D: BUFFER_TYPE = {}

        o, info = env.reset()
        R, R_arr = 0, []
        for i in range(N):
            a_dic, logp, V = s.get_action(o)
            a = a_dic["action"]
            o_next, r, done, truncated, info = env.step(a)
            R += r
            V_next = 0
            ended = done or truncated or i == N - 1
            if ended:
                if not done:
                    V_next = s.get_action(o_next)[-1]
                o_next, info = env.reset()
                R_arr.append(R)
                R = 0
            # requires: r, ended, V, V_next, o, a, logp
            dic = dict(r=r, ended=ended, V=V, V_next=V_next, o=o, a=a, logp=logp)
            add_record(D, i, dic, N)
            o = o_next
        print(f"mean R: {np.mean(R_arr)}")
        D["a_dic"] = {"action": D["a"]}
        return D
