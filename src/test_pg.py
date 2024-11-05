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
