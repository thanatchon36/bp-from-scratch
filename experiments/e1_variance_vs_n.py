"""E1 / E4 — Var[dC/dtheta] of the McClean random-axis HEA versus n and depth L.

Cost operator H = Z^{(x)n}.  For every (n, L) we draw ``samples`` random circuits
(axes and angles), compute the full gradient with the adjoint method, and record the
variance of the derivative w.r.t. the first, middle and last rotation, the mean
variance over all parameters, and the half-chain entanglement entropy of U|0> (E4).

    python experiments/e1_variance_vs_n.py --n-list 2 4 6 8 10 12 --L-list 2 5 10 20 50
    python experiments/e1_variance_vs_n.py --exp e4 --n-list 12 --L-list 1 2 3 4 6 8 12 16 24 32 50
"""
import argparse

import numpy as np

from common import out_of_time, haar_variance_two_sided, load, log, sample_gradients, save
from ansatz.layered import hea_mcclean, probe_indices
from qsim import PauliSum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e1")
    ap.add_argument("--n-list", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 14, 16])
    ap.add_argument("--L-list", type=int, nargs="+", default=[2, 5, 10, 20, 50])
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for n in args.n_list:
        for L in args.L_list:
            key = f"n{n:02d}_L{L:03d}"
            if load(args.exp, key):
                continue
            if out_of_time():
                return
            rng = np.random.default_rng(args.seed + 1000 * n + L)
            H = PauliSum.global_z(n)
            res = sample_gradients(lambda r: hea_mcclean(n, L, r), H, args.samples, rng,
                                   keep=probe_indices(n, L), entropy_cut=n // 2)
            res.update({"n": n, "L": L, "ham": "globalZ",
                        "haar_two_sided": haar_variance_two_sided(2**n, 0.0, 2.0**n)})
            save(args.exp, key, res)
            log(f"{args.exp} n={n:2d} L={L:3d}  var_mid={res['var_mid']:.3e}  haar={res['haar_two_sided']:.3e}  "
                f"var_mean={res['var_mean_over_params']:.3e}  S={res['entropy_mean']:.3f}  ({res['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
