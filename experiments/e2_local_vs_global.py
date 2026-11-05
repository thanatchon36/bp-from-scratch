"""E2 — cost-function dependence (Cerezo et al. 2021): local vs global cost on the
alternating-layered ansatz (RY-RZ rotations, CZ bricks).

Global cost  C_G = 1 - <0...0|rho|0...0>           -> gradient operator  |0..0><0..0|
Local  cost  C_L = 1 - (1/n) sum_i <0|rho_i|0>      -> gradient operator  (1/2n) sum_i Z_i
(constants and signs do not change the variance).
"""
import argparse

import numpy as np

from common import out_of_time, load, log, sample_gradients, save
from ansatz.layered import ala
from qsim import PauliSum, Projector0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    ap.add_argument("--L-list", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--costs", nargs="+", default=["global", "local"])
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for cost in args.costs:
        for n in args.n_list:
            for L in args.L_list:
                key = f"{cost}_n{n:02d}_L{L:02d}"
                if load("e2", key):
                    continue
                if out_of_time():
                    return
                rng = np.random.default_rng(args.seed + 1000 * n + L + (0 if cost == "global" else 7))
                H = Projector0(n) if cost == "global" else PauliSum.local_z(n) * 0.5
                keep = {"first": 0, "mid": (L // 2) * 2 * n + 2 * (n // 2)}
                res = sample_gradients(lambda r: ala(n, L), H, args.samples, rng, keep=keep)
                res.update({"n": n, "L": L, "cost": cost})
                save("e2", key, res)
                log(f"e2 {cost:6s} n={n:2d} L={L:2d}  var_first={res['var_first']:.3e}  var_mid={res['var_mid']:.3e}  "
                    f"var_mean={res['var_mean_over_params']:.3e}  ({res['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
