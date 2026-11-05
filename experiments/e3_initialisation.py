"""E3 — gradient variance at initialisation for different initialisation strategies.

Ansatz: RY-RZ hardware-efficient ansatz with a CZ ladder, 20 CZ layers in every variant.
  uniform        theta ~ U[0, 2pi)
  gauss_<sigma>  theta ~ N(0, sigma^2)                     (Zhang et al. 2022)
  identity_block M = 10 identity blocks, Grant et al. 2019 (8nM parameters, 2M CZ layers)
Cost operator: the local observable  H = (1/n) sum_i X_i  (its optimum is not at theta = 0,
so small angles are not trivially at a stationary point).
"""
import argparse

import numpy as np

from common import out_of_time, load, log, sample_gradients, save
from ansatz.layered import hea_ryrz, identity_block_hea
from qsim import PauliSum


def local_x(n):
    return PauliSum([(1.0 / n, "I" * i + "X" + "I" * (n - i - 1)) for i in range(n)], n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    ap.add_argument("--variants", nargs="+", default=["uniform", "gauss_1.0", "gauss_0.3", "gauss_0.1", "identity_block"])
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    L = args.L
    for variant in args.variants:
        for n in args.n_list:
            key = f"{variant}_n{n:02d}"
            if load("e3", key):
                continue
            if out_of_time():
                return
            rng = np.random.default_rng(args.seed + 1000 * n + hash(variant) % 997)
            H = local_x(n)
            keep = {"first": 0, "mid": (L // 2) * 2 * n + 2 * (n // 2)}
            if variant == "identity_block":
                M = L // 2

                def build(r):
                    c, init = identity_block_hea(n, M, r)
                    c.init_params = init  # type: ignore[attr-defined]
                    return c

                sampler = lambda c, r: c.init_params  # noqa: E731
                keep = {"first": 0, "mid": (M // 2) * 8 * n + 2 * (n // 2)}
            else:
                build = lambda r: hea_ryrz(n, L)  # noqa: E731
                if variant == "uniform":
                    sampler = None
                else:
                    sigma = float(variant.split("_")[1])
                    sampler = lambda c, r, s=sigma: r.normal(0.0, s, size=c.n_params)  # noqa: E731
            res = sample_gradients(build, H, args.samples, rng, param_sampler=sampler, keep=keep)
            res.update({"n": n, "L": L, "variant": variant})
            save("e3", key, res)
            log(f"e3 {variant:15s} n={n:2d}  var_first={res['var_first']:.3e}  var_mid={res['var_mid']:.3e}  "
                f"var_mean={res['var_mean_over_params']:.3e}  ({res['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
