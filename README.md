# minRL: Deep Reinforcement Learning for `minimalists`

```cmd
pip install minRL
```

```py
import gymnasium as gym

from minRL.utils.nn_utils import mlp
from minRL.vpg.VanillaPG import VanillaPG

env = gym.make("CartPole-v1")
a_space = env.action_space
discrete = isinstance(a_space, gym.spaces.Discrete)
a_dim = a_space.n if discrete else a_space.shape[0]
o_dim = env.observation_space.shape[0]
pi_net = mlp([o_dim, 64, 64, a_dim])
V_net = mlp([o_dim, 64, 64, 1])

pg = VanillaPG(discrete, a_dim, pi_net, V_net, pi_lr=2e-3)
for e in range(500):
    pg.train_once(pg.get_D_from_env(env))

```

## Intro

- I am **starting** to learn RL following [OpenAI SpinningUp](https://github.com/openai/spinningup)

- As I dig into [the code](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/vpg), I find it quite **verbose**, and found [some confusing places (maybe bugs?)](https://github.com/openai/spinningup/issues/424)

- Coming from a Physics background, I love **minimalism** and **simplicity**... hence I made this repo

## Example comparison: Vanilla Policy Gradient

- `VPG` by OpenAI SpinningUp:
    - [135 lines @ core.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py)
        - Contains `combined_shape, mlp, count_vars, discount_cumsum, Actor, MLPCategoricalActor, MLPGaussianActor, MLPCritic, MLPActorCritic`
    - [350 lines @ vpg.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py)
        - Contains `VPGBuffer, vpg`
    - Highly nested classes: `actor-critic -> actor & critic -> mlp`
    - Key equations scattered all over the place
    - Highly coupled with `gym` interface

- `VPG` I wrote:
    - [27 lines @ nn_utils.py](./src/minRL/utils/nn_utils.py)
        - Contains `discount_cum_sum, tensor, mlp`
    - [152 lines @ VanillaPG.py](./src/minRL/vpg/VanillaPG.py)
        - Contains `VanillaPG`
    - No highly nested classes
    - All key equations in a single `53 lines @ VanillaPG.train_once(D)` method, following the [pseudo-code below](https://spinningup.openai.com/en/latest/algorithms/vpg.html#pseudocode)
    - **Decoupled** from the `gym` interface
        - This is the most important feature for the RL problem I am interested in, as it avoids back-and-forth interaction with the environment from the RL code, but instead only interacts with Replay Buffer data `D`
        - An optional util function `VanillaPG.get_D_from_env` is provided 
    - Correctness verified by [test_vpg.py](./src/test_vpg.py)

![](https://spinningup.openai.com/en/latest/_images/math/262538f3077a7be8ce89066abbab523575132996.svg)

```py
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
```
