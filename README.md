# minRL: Deep Reinforcement Learning for `minimalists`

```cmd
pip install minRL
```

```py
import gymnasium as gym

import minRL.utils.mpi_fake as mpi
from minRL.ppo.PPOClip import PPOClip
from minRL.utils.nn_utils import mlp
from minRL.vpg.MixedActor import MixedActor
from minRL.vpg.VanillaPG import VanillaPG

mpi.start_workers()

env = gym.make("CartPole-v1")
a_space = env.action_space
discrete = isinstance(a_space, gym.spaces.Discrete)
a_dim = a_space.n if discrete else a_space.shape[0]
o_dim = env.observation_space.shape[0]

nets = {
    "shared": {"net": mlp([o_dim, 64, 64])},
    "action": {"net": mlp([64, a_dim]), "values": range(a_dim)},
}
pi_net = MixedActor(nets)
V_net = mlp([o_dim, 64, 64, 1])

UsedPG = VanillaPG if 0 else PPOClip

pg = UsedPG(pi_net, V_net, pi_lr=2e-3)
for e in range(100):
    pg.train_once(pg.get_D_from_env(env))

```

## Intro

- I am **starting** (`2024-10-19`) to learn RL following [OpenAI SpinningUp](https://github.com/openai/spinningup)

- As I dig into [the code](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/vpg), I find it quite **verbose**, and found [some confusing places (maybe bugs?)](https://github.com/openai/spinningup/issues/424)

- Coming from a Physics background, I love **minimalism** and **simplicity**... hence I made this repo

## Example comparison: Vanilla Policy Gradient

- `VPG` by OpenAI SpinningUp:
    - [135 lines @ core.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py)
        - Contains `combined_shape, mlp, count_vars, discount_cumsum, Actor, MLPCategoricalActor, MLPGaussianActor, MLPCritic, MLPActorCritic`
    - [350 lines @ vpg.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py)
        - Contains `VPGBuffer, vpg`
    - Highly nested classes: `actor-critic -> actor & critic -> mlp`
    - Neural Network models coupled with RL algorithms
    - Key equations scattered all over the place
    - Highly coupled with `gym` interface
    - Only supports either `Box (Normal)` or `Discrete (Categorical)` action spaces, not arbitrary mixture of them

- `VPG` I wrote:
    - [30 lines @ nn_utils.py](./src/minRL/utils/nn_utils.py)
        - Contains `discount_cum_sum, tensor, mlp`
    - [47 lines @ MixedActor.py](./src/minRL/vpg/MixedActor.py)
        - Contains `MixedActor`
    - [130 lines @ VanillaPG.py](./src/minRL/vpg/VanillaPG.py)
        - Contains `VanillaPG, add_record`
    - No highly nested classes
    - Neural Network models **decoupled** from RL algorithms
    - All key equations in one place, following the [pseudo-code below](https://spinningup.openai.com/en/latest/algorithms/vpg.html#pseudocode)
        ![](https://spinningup.openai.com/en/latest/_images/math/262538f3077a7be8ce89066abbab523575132996.svg)
    - **Decoupled** from the `gym` interface
        - This is the most important feature for the RL problem I am interested in, as it avoids back-and-forth interaction with the environment from the RL code, but instead only interacts with Replay Buffer data `D`
        - An optional util function `VanillaPG.get_D_from_env` is provided 
    - Correctness verified by [test_pg.py](./src/test_pg.py)
    - Supports arbitrary mixture of `Box (Normal)` and `Discrete (Categorical)` actions

```py
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
```

```py
class VanillaPG:
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
```

## Example comparison: PPO Clip

- `PPO Clip` by OpenAI SpinningUp:
    - [135 lines @ core.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py)
    - [378 lines @ ppo.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)
- `PPO Clip` I wrote:
    - [32 lines @ PPOClip.py](./src/minRL/ppo/PPOClip.py)

![](https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg)

```py
class PPOClip(VanillaPG):
    pi_iters = 80
    eps = 0.2
    max_kl = 0.015

    def update_pi(s, D: BUFFER_TYPE):
        o, a_dic, A, logp_old = D["o"], D["a_dic"], D["A"], D["logp"]
        for _ in range(s.pi_iters):
            s.pi_opt.zero_grad()
            pi_dic = s.pi_net.get_pi(o)
            logp = s.pi_net.get_logp(pi_dic, a_dic)

            ratio = tc.exp(logp - logp_old)
            r_clip = tc.clamp(ratio, 1 - s.eps, 1 + s.eps)
            loss = -(tc.min(ratio * A, r_clip * A)).mean()

            kl = mpi.avg((logp_old - logp).mean().item())
            if kl > s.max_kl:
                print("kl > max_kl, stopping!")
                break
            # ent = pi.entropy().mean().item()
            # clipped = ratio.gt(1 + s.eps) | ratio.lt(1 - s.eps)
            # clip_frac = tensor(clipped).mean().item()

            loss.backward()
            mpi.avg_grads(s.pi_params)
            s.pi_opt.step()
```
