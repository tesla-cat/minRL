# cspell:ignore logp
from typing import Dict

import gymnasium as gym
import numpy as np
import torch as tc
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam

import minRL.utils.mpi_fake as mpi
from minRL.utils.nn_utils import discount_cum_sum, tensor

BUFFER_TYPE = Dict[str, np.ndarray]


class VanillaPG:
    def __init__(
        s,
        discrete,
        a_dim,
        pi_net: nn.Module,
        V_net: nn.Module,
        seed=0,
        pi_lr=3e-4,
        V_lr=1e-3,
    ):
        mpi.start_workers()
        mpi.setup_torch()

        s.seed = seed + 1000 * mpi.proc_id()
        tc.manual_seed(s.seed)
        np.random.seed(s.seed)

        s.discrete = discrete
        s.log_std = nn.Parameter(tensor(np.full(a_dim, -0.5)))
        s.pi_net, s.V_net = pi_net, V_net

        s.pi_params = list(s.pi_net.parameters()) + [s.log_std]
        s.V_params = list(s.V_net.parameters())
        s.pi_opt = Adam(s.pi_params, lr=pi_lr)
        s.V_opt = Adam(s.V_params, lr=V_lr)
        mpi.sync_params(s.pi_params + s.V_params)

    def train_once(
        s,
        D: BUFFER_TYPE,
        gam=0.99,
        lam=0.97,
        V_iters=80,
    ):
        # =====================================
        # 4, 5: compute R and A (requires: r, ended, V, V_next, o, a, logp)

        for k in ["R", "A"]:
            D[k] = np.zeros_like(D["r"])
        start = 0
        N = len(D["r"])
        for i in range(N):
            if D["ended"][i]:
                V_next = D["V_next"][i]
                slc = slice(start, i + 1)
                r = np.append(D["r"][slc], V_next)
                V = np.append(D["V"][slc], V_next)
                # GAE-lambda advantage estimation
                A = r[:-1] + gam * V[1:] - V[:-1]
                D["A"][slc] = discount_cum_sum(A, gam * lam)
                D["R"][slc] = discount_cum_sum(r, gam)[:-1]
                start = i + 1
        D["A"] = mpi.normalize(D["A"])
        D = {k: tensor(v) for k, v in D.items()}

        # ======================================
        # 6, 7: estimate pg and optimize
        o, a, A, R, logp_old = D["o"], D["a"], D["A"], D["R"], D["logp"]

        s.pi_opt.zero_grad()
        pi = s.get_pi(o)
        logp = s.get_logp(pi, a)
        pi_loss = -(logp * A).mean()
        if 0:
            # shouldn't logp_old and logp be the same as pi is not updated yet?
            kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            print(f"kl: {kl}, ent: {ent}")
        pi_loss.backward()
        mpi.avg_grads(s.pi_params)
        s.pi_opt.step()

        # =============================
        # 8: fit V

        for i in range(V_iters):
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
