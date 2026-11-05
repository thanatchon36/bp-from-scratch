"""E5 — training in practice: prepare |0...0> with the alternating-layered ansatz from a
uniformly random initialisation, minimising either the global cost C_G = 1 - <P_0> or the
local cost C_L = 1 - (1/n) sum_i <P_0^(i)>.  We record the fidelity reached, the cost
history and the gradient norm at initialisation for every seed.
"""
import argparse

import numpy as np

from common import out_of_time, adam, load, log, save
from ansatz.layered import ala
from qsim import PauliSum, Projector0, adjoint_gradient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--costs", nargs="+", default=["global", "local"])
    ap.add_argument("--shots", type=int, default=0,
                    help="if > 0, add Gaussian shot noise of std 1/sqrt(2*shots) to every cost and gradient entry")
    args = ap.parse_args()
    sigma = 1.0 / np.sqrt(2.0 * args.shots) if args.shots > 0 else 0.0
    suffix = f"_shots{args.shots}" if args.shots > 0 else ""
    for cost in args.costs:
        for n in args.n_list:
            key = f"{cost}_n{n:02d}{suffix}"
            if load("e5", key):
                continue
            if out_of_time():
                return
            c = ala(n, args.L)
            P0 = Projector0(n)
            Hl = PauliSum.local_z(n) * 0.5  # C_L = 1/2 - <Hl>
            runs = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(10_000 * seed + n + (0 if cost == "global" else 1))
                x0 = c.random_params(rng)
                noise = np.random.default_rng(99_000 + 10_000 * seed + n)

                def cg(x):
                    if cost == "global":
                        val, g = adjoint_gradient(c, x, P0, return_cost=True)
                        val, g = 1.0 - val, -g
                    else:
                        val, g = adjoint_gradient(c, x, Hl, return_cost=True)
                        val, g = 0.5 - val, -g
                    if sigma > 0:
                        val = val + noise.normal(0.0, sigma)
                        g = g + noise.normal(0.0, sigma, size=g.shape)
                    return val, g

                c0, g0 = cg(x0)
                x, hist = adam(cg, x0, args.steps, lr=args.lr)
                fid = P0.expectation(c.run(x))
                runs.append({"seed": seed, "cost_init": float(c0), "grad_norm_init": float(np.linalg.norm(g0)),
                             "cost_final": float(hist[-1]), "fidelity_final": float(fid),
                             "history": [float(h) for h in hist[::5]]})
            fids = np.array([r["fidelity_final"] for r in runs])
            out = {"n": n, "L": args.L, "cost": cost, "steps": args.steps, "lr": args.lr, "runs": runs,
                   "shots": args.shots, "sigma": sigma,
                   "fidelity_mean": float(fids.mean()), "fidelity_std": float(fids.std()),
                   "success_rate": float(np.mean(fids > 0.99))}
            save("e5", key, out)
            log(f"e5 {cost:6s}{suffix} n={n:2d}: fidelity {out['fidelity_mean']:.3f} +- {out['fidelity_std']:.3f}  "
                f"success(F>0.99)={out['success_rate']:.2f}  |g0|~{np.mean([r['grad_norm_init'] for r in runs]):.2e}")


if __name__ == "__main__":
    main()
