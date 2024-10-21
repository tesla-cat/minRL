# cspell:ignore logp
import minRL.utils.mpi_fake as mpi
from minRL.vpg.VanillaPG import BUFFER_TYPE, VanillaPG, tc


class PPOClip(VanillaPG):
    pi_iters = 80
    eps = 0.2
    max_kl = 0.015

    def update_pi(s, D: BUFFER_TYPE):
        o, a, A, logp_old = D["o"], D["a"], D["A"], D["logp"]
        for _ in range(s.pi_iters):
            s.pi_opt.zero_grad()
            pi = s.get_pi(o)
            logp = s.get_logp(pi, a)

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
