import gymnasium as gym

from minRL.ppo.PPOClip import PPOClip
from minRL.utils.nn_utils import mlp
from minRL.vpg.VanillaPG import VanillaPG

env = gym.make("CartPole-v1")
a_space = env.action_space
discrete = isinstance(a_space, gym.spaces.Discrete)
a_dim = a_space.n if discrete else a_space.shape[0]
o_dim = env.observation_space.shape[0]
pi_net = mlp([o_dim, 64, 64, a_dim])
V_net = mlp([o_dim, 64, 64, 1])

UsedPG = VanillaPG if 0 else PPOClip

pg = UsedPG(discrete, a_dim, pi_net, V_net, pi_lr=2e-3)
for e in range(100):
    pg.train_once(pg.get_D_from_env(env))
