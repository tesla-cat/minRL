# cspell:ignore logp
from typing import Dict, Union

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam

import minRL.utils.mpi_fake as mpi
from minRL.utils.nn_utils import discount_cum_sum, tensor

BUFFER_TYPE = Dict[str, Union[np.ndarray, tc.Tensor]]


class VanillaPG:
    def __init__(
        s,
        discrete,
        a_dim,
        pi_net: nn.Module,
        V_net: nn.Module,
        pi_lr=3e-4,
        V_lr=1e-3,
        V_iters=80,
        gam=0.99,
        lam=0.97,
        seed=0,
    ):
        mpi.start_workers()
        mpi.setup_torch()

        s.seed = seed + 1000 * mpi.proc_id()
        tc.manual_seed(s.seed)
        np.random.seed(s.seed)

        s.discrete = discrete
        s.pi_net, s.V_net = pi_net, V_net
        s.V_iters = V_iters
        s.gam, s.lam = gam, lam

        s.log_std = nn.Parameter(tensor(np.full(a_dim, -0.5)))
        s.pi_params = list(s.pi_net.parameters()) + [s.log_std]
        s.V_params = list(s.V_net.parameters())
        s.pi_opt = Adam(s.pi_params, lr=pi_lr)
        s.V_opt = Adam(s.V_params, lr=V_lr)
        mpi.sync_params(s.pi_params + s.V_params)

    def train_once(s, D: BUFFER_TYPE):
        # requires: r, ended, V, V_next, o, a, logp
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
        return {k: tensor(v) for k, v in D.items()}

    def update_pi(s, D: BUFFER_TYPE):
        # 6, 7: estimate pg and optimize
        o, a, A = D["o"], D["a"], D["A"]
        s.pi_opt.zero_grad()
        pi = s.get_pi(o)
        logp = s.get_logp(pi, a)
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

    # ============================

    def get_pi(s, o):
        if s.discrete:
            return Categorical(logits=s.pi_net(o))
        else:
            return Normal(s.pi_net(o), tc.exp(s.log_std))

    def get_logp(s, pi: Normal, a) -> tc.Tensor:
        return pi.log_prob(a) if s.discrete else pi.log_prob(a).sum(-1)

    def get_V(s, o):
        return tc.squeeze(s.V_net(o), -1)

    # ============================================

    def get_action(s, o):
        with tc.no_grad():
            o = tensor(o)
            pi = s.get_pi(o)
            a = pi.sample()
            return a.numpy(), s.get_V(o).numpy(), s.get_logp(pi, a).numpy()

    def get_D_from_env(s, env: gym.Env, N=4000):
        D: BUFFER_TYPE = {}

        def add(i, dic: Dict):
            for k, v in dic.items():
                if k not in D:
                    shape = v.shape if isinstance(v, np.ndarray) else ()
                    D[k] = np.zeros((N, *shape), np.float32)
                D[k][i] = v

        o, info = env.reset(seed=s.seed)
        R, R_arr = 0, []
        for i in range(N):
            a, V, logp = s.get_action(o)
            o_next, r, done, truncated, info = env.step(a)
            R += r
            V_next = 0
            ended = done or truncated or i == N - 1
            if ended:
                if not done:
                    _, V_next, _ = s.get_action(o_next)
                o_next, info = env.reset(seed=s.seed)
                R_arr.append(R)
                R = 0
            # requires: r, ended, V, V_next, o, a, logp
            add(i, dict(r=r, ended=ended, V=V, V_next=V_next, o=o, a=a, logp=logp))
            o = o_next
        print(f"mean R: {np.mean(R_arr)}")
        return D
