"""E7 — Lie-algebraic prediction  Var[C] = P_g(rho) P_g(H) / dim(g)  (Ragone et al. 2024).

Four fully-parameterised ansatz families (all gates are Pauli rotations):
  hea_real   generators {Y_i, Z_i Z_{i+1}},           H = sum_i Z_i Z_{i+1},  rho = |0><0|^n
  hea_full   generators {X_i, Y_i, Z_i Z_{i+1}},      H = sum_i Z_i Z_{i+1},  rho = |0><0|^n
  hva_tfim   generators {sum_i X_i, sum_i Z_i Z_{i+1}}, H = sum_i Z_i Z_{i+1}, rho = |0><0|^n
  qaoa       generators {sum_E Z_i Z_j, sum_i X_i} on a random 3-regular graph,
             H = sum_E Z_i Z_j (equivalently the cut operator),  rho = |+><+|^n
For each we compute dim(g) and the g-purities exactly, and measure Var[C] over uniformly
random parameters of a deep (L-layer) circuit.
"""
import argparse
import time

import numpy as np

from common import out_of_time, load, log, sample_costs, save
from ansatz.layered import pauli_hea
from ansatz.qaoa import hva_tfim, qaoa_maxcut, random_regular_graph, zz_chain
from qsim import PauliSum
from qsim.dla import (LieClosure, label_to_pauli, operator_from_terms, pauli_string_closure,
                      product_state_operator, set_purity)


def single(n, q, letter):
    return "I" * q + letter + "I" * (n - q - 1)


def bond(n, i, j, letter="Z"):
    s = ["I"] * n
    s[i] = s[j] = letter
    return "".join(s)


def dla_for(family, n, edges=None, cap=4000):
    """Returns (dim, purity_fn) where purity_fn(op) = P_g(op)."""
    if family in ("hea_real", "hea_full"):
        gens = [label_to_pauli(single(n, q, "Y")) for q in range(n)]
        gens += [label_to_pauli(bond(n, i, i + 1)) for i in range(n - 1)]
        if family == "hea_full":
            gens += [label_to_pauli(single(n, q, "X")) for q in range(n)]
        strings = pauli_string_closure(gens)
        return len(strings), (lambda op: set_purity(strings, op, n)), False
    if family == "hva_tfim":
        gens = [operator_from_terms([(1.0, single(n, q, "X")) for q in range(n)]),
                operator_from_terms([(1.0, bond(n, i, i + 1)) for i in range(n - 1)])]
    else:
        gens = [operator_from_terms([(1.0, bond(n, i, j)) for i, j in edges]),
                operator_from_terms([(1.0, single(n, q, "X")) for q in range(n)])]
    lc = LieClosure(gens, cap=cap)
    return lc.dim, (lambda op: lc.purity(op, n)), lc.capped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["hea_real", "hea_full", "hva_tfim", "qaoa"])
    ap.add_argument("--n-max", type=int, nargs="+", default=[8, 7, 10, 6], help="per family")
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--samples", type=int, default=400)
    args = ap.parse_args()
    for family, nmax in zip(args.families, args.n_max):
        for n in range(2, nmax + 1):
            if family == "qaoa" and (n < 4 or n % 2):
                continue
            key = f"{family}_n{n:02d}"
            if load("e7", key):
                continue
            if out_of_time():
                return
            rng = np.random.default_rng(777 + n)
            edges = random_regular_graph(n, 3, rng) if family == "qaoa" else None
            t0 = time.perf_counter()
            dim, purity, capped = dla_for(family, n, edges)
            t_dla = time.perf_counter() - t0
            if family == "qaoa":
                H = PauliSum([(1.0, bond(n, i, j)) for i, j in edges], n=n)
                rho = product_state_operator(n, "+")
                build = lambda r: qaoa_maxcut(n, edges, args.L)  # noqa: E731
            else:
                H = zz_chain(n)
                rho = product_state_operator(n, "0")
                if family == "hva_tfim":
                    build = lambda r: hva_tfim(n, args.L)  # noqa: E731
                else:
                    build = lambda r, f=(family == "hea_full"): pauli_hea(n, args.L, full=f)  # noqa: E731
            Hop = operator_from_terms(H.terms)
            p_rho, p_h = purity(rho), purity(Hop)
            pred = p_rho * p_h / dim if not capped else float("nan")
            meas = sample_costs(build, H, args.samples, rng)
            res = {"family": family, "n": n, "dim": dim, "capped": capped, "purity_rho": p_rho, "purity_H": p_h,
                   "predicted_var": pred, "measured_var": meas["cost_var"], "measured_var_ci": meas["cost_var_ci"],
                   "cost_mean": meas["cost_mean"], "L": args.L, "samples": args.samples, "seconds_dla": t_dla,
                   "edges": edges, "full_dim": 4**n - 1}
            save("e7", key, res)
            log(f"e7 {family:9s} n={n}: dim(g)={dim}{'+' if capped else ''} (su: {4**n-1})  P_rho={p_rho:.3f} "
                f"P_H={p_h:.1f}  pred={pred:.3e}  meas={meas['cost_var']:.3e}  (dla {t_dla:.1f}s)")


if __name__ == "__main__":
    main()
