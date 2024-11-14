from minRL.ppo import PPOClip, Recorder, gym, make_actor_critic, np, tc
from minRL.spin_ppo import ppo as spin_ppo

rec = Recorder()

for id in [
    "Acrobot-v1",
    "CartPole-v1",
    "MountainCarContinuous-v0",
    "MountainCar-v0",
    "Pendulum-v1",
]:
    print(id)
    tc.manual_seed(0)
    np.random.seed(0)
    env = gym.make(id)
    pi_net, V_net = make_actor_critic(env)
    ppo = PPOClip(pi_net, V_net)
    for e in range(50):
        loss_pi, loss_V = ppo.learn(ppo.get_D_from_env(env, rec, N=4000, max_L=1000))
    print(loss_pi, loss_V)

    # spin-up ppo
    spin_ppo(
        lambda: gym.make(id), rec, epochs=50, steps_per_epoch=4000, max_ep_len=1000
    )

    rec.save()
