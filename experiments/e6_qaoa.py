"""E6 — problem-inspired ansatz: QAOA for MaxCut on random 3-regular graphs.

(a) Variance of dC/dgamma_1 and dC/dbeta_1 over random graphs and uniform parameters,
    versus n for several depths p (compare with the HEA slopes of E1).
(b) Approximation ratio <C>/C_max reached by Adam from random starts, versus p (n = 10).
"""
import argparse

import numpy as np

from common import out_of_time, adam, load, log, sample_gradients, save
from ansatz.qaoa import maxcut_value, qaoa_maxcut, random_regular_graph
from qsim import PauliSum, adjoint_gradient


def variance_part(args):
    for p in args.p_list:
        for n in args.n_list:
            key = f"var_n{n:02d}_p{p:02d}"
            if load("e6", key):
                continue
            if out_of_time():
                return
            rng = np.random.default_rng(1000 * n + p)
            edges_holder = {}

            def build(r):
                edges = random_regular_graph(n, 3, r)
                edges_holder["edges"] = edges
                return qaoa_maxcut(n, edges, p)

            # the Hamiltonian depends on the graph -> wrap sample_gradients per sample
            raw_g, raw_b, costs = [], [], []
            import time
            t0 = time.perf_counter()
            for _ in range(args.samples):
                c = build(rng)
                H = PauliSum.maxcut(n, edges_holder["edges"])
                params = c.random_params(rng)
                cost, g = adjoint_gradient(c, params, H, return_cost=True)
                raw_g.append(float(g[0]))
                raw_b.append(float(g[1]))
                costs.append(cost)
            from common import bootstrap_var_ci
            res = {"n": n, "p": p, "n_samples": args.samples, "seconds": time.perf_counter() - t0,
                   "var_gamma1": float(np.var(raw_g)), "var_gamma1_ci": list(bootstrap_var_ci(raw_g)),
                   "var_beta1": float(np.var(raw_b)), "var_beta1_ci": list(bootstrap_var_ci(raw_b)),
                   "cost_mean": float(np.mean(costs)), "cost_var": float(np.var(costs)),
                   "raw_gamma1": raw_g, "raw_beta1": raw_b}
            save("e6", key, res)
            log(f"e6 var n={n:2d} p={p}: var_gamma1={res['var_gamma1']:.3e} var_beta1={res['var_beta1']:.3e} "
                f"({res['seconds']:.1f}s)")


def ratio_part(args):
    n = args.ratio_n
    for p in args.ratio_p_list:
        key = f"ratio_n{n:02d}_p{p:02d}"
        if load("e6", key):
            continue
        if out_of_time():
            return
        out = {"n": n, "p": p, "graphs": []}
        for gi in range(args.graphs):
            rng = np.random.default_rng(500 + gi)
            edges = random_regular_graph(n, 3, rng)
            H = PauliSum.maxcut(n, edges)
            cmax = maxcut_value(n, edges)
            c = qaoa_maxcut(n, edges, p)
            best = -1.0
            inits = []
            for ii in range(args.inits):
                x0 = rng.uniform(0, 2 * np.pi, size=c.n_params)

                def cg(x):
                    val, g = adjoint_gradient(c, x, H, return_cost=True)
                    return -val, -g  # maximise the cut

                x, hist = adam(cg, x0, args.steps, lr=args.lr)
                ratio = -hist[-1] / cmax
                inits.append(float(ratio))
                best = max(best, ratio)
            out["graphs"].append({"edges": edges, "cmax": cmax, "ratios": inits, "best": float(best)})
        out["best_mean"] = float(np.mean([g["best"] for g in out["graphs"]]))
        out["all_mean"] = float(np.mean([r for g in out["graphs"] for r in g["ratios"]]))
        save("e6", key, out)
        log(f"e6 ratio n={n} p={p}: best-of-inits {out['best_mean']:.3f}, mean over inits {out['all_mean']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["var", "ratio", "both"], default="both")
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14, 16])
    ap.add_argument("--p-list", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--ratio-n", type=int, default=10)
    ap.add_argument("--ratio-p-list", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--graphs", type=int, default=3)
    ap.add_argument("--inits", type=int, default=4)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()
    if args.part in ("var", "both"):
        variance_part(args)
    if args.part in ("ratio", "both"):
        ratio_part(args)


if __name__ == "__main__":
    main()
