"""Shared utilities for the experiments: caching, sampling, statistics, optimisation."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

from qsim import Statevector, adjoint_gradient  # noqa: E402


# ----------------------------------------------------------------------- caching
def result_path(exp: str, key: str) -> Path:
    return RESULTS / exp / f"{key}.json"


def load(exp: str, key: str):
    p = result_path(exp, key)
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return None


def save(exp: str, key: str, obj) -> None:
    p = result_path(exp, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(obj, fh)


def load_all(exp: str):
    d = RESULTS / exp
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        with open(p) as fh:
            out.append(json.load(fh))
    return out


# ----------------------------------------------------------------------- statistics
def bootstrap_var_ci(x, n_boot: int = 400, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the sample variance."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    v = np.var(x[idx], axis=1)
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


def fit_log2_slope(ns, values, weights=None):
    """Least-squares slope of log2(values) vs n (decay exponent)."""
    ns = np.asarray(ns, float)
    y = np.log2(np.asarray(values, float))
    A = np.vstack([ns, np.ones_like(ns)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


# ----------------------------------------------------------------------- closed-form Haar references
def haar_variance_two_sided(d: int, trH: float, trH2: float) -> float:
    """Var[dC/dtheta] when the circuit before AND after the rotation are independent Haar
    unitaries (unitary 2-designs), for any Pauli generator G:
        Var = [d Tr(H^2) - Tr(H)^2] / (2 (d+1)^2 (d-1)).      (derived in the appendix)"""
    return (d * trH2 - trH**2) / (2.0 * (d + 1) ** 2 * (d - 1))


def haar_variance_left_only(d: int, trH: float, trH2: float, trK2: float) -> float:
    """Var when only the circuit AFTER the rotation is Haar; the state before it is fixed with
    K = i[sigma, G],  Tr(K^2) = 2 - 2 <G>_sigma^2:
        Var = Tr(K^2) [d Tr(H^2) - Tr(H)^2] / (4 d (d^2 - 1))."""
    return trK2 * (d * trH2 - trH**2) / (4.0 * d * (d**2 - 1))


def haar_variance_right_only(d: int, trH2: float, trGHGH: float) -> float:
    """Var when only the circuit BEFORE the rotation is Haar and H is fixed:
        Var = [2 Tr(H^2) - 2 Tr(G H G H)] / (4 d (d+1))."""
    return (2.0 * trH2 - 2.0 * trGHGH) / (4.0 * d * (d + 1))


# ----------------------------------------------------------------------- sampling
def sample_gradients(build: Callable[[np.random.Generator], object], ham, n_samples: int,
                     rng: np.random.Generator, param_sampler: Optional[Callable] = None,
                     keep: Optional[dict] = None, entropy_cut: Optional[int] = None):
    """Draw ``n_samples`` random circuits/parameters, return gradient statistics.

    build(rng) -> Circuit (may itself be random, e.g. random rotation axes)
    param_sampler(circuit, rng) -> parameter vector (default: uniform [0, 2pi))
    keep: {name: parameter index} whose raw derivatives are returned for bootstrapping
    entropy_cut: if given, also record the entanglement entropy of qubits [0, cut) of U|0>.
    """
    keep = keep or {}
    kept = {k: [] for k in keep}
    costs, ents = [], []
    sum_g = None
    sum_g2 = None
    t0 = time.perf_counter()
    for _ in range(n_samples):
        c = build(rng)
        params = c.random_params(rng) if param_sampler is None else param_sampler(c, rng)
        cost, g = adjoint_gradient(c, params, ham, return_cost=True)
        costs.append(cost)
        if sum_g is None:
            sum_g = np.zeros_like(g)
            sum_g2 = np.zeros_like(g)
        sum_g += g
        sum_g2 += g * g
        for k, idx in keep.items():
            kept[k].append(float(g[idx]))
        if entropy_cut is not None:
            ents.append(c.run(params).entanglement_entropy(range(entropy_cut)))
    N = n_samples
    mean_g = sum_g / N
    var_g = sum_g2 / N - mean_g**2  # per-parameter population variance
    out = {
        "n_samples": N,
        "n_params": int(len(mean_g)),
        "var_mean_over_params": float(np.mean(var_g)),
        "var_median_over_params": float(np.median(var_g)),
        "var_per_param": [float(v) for v in var_g],
        "grad_mean_abs": float(np.mean(np.abs(mean_g))),
        "cost_mean": float(np.mean(costs)),
        "cost_var": float(np.var(costs)),
        "seconds": time.perf_counter() - t0,
    }
    for k in keep:
        arr = np.asarray(kept[k])
        lo, hi = bootstrap_var_ci(arr)
        out[f"var_{k}"] = float(np.var(arr))
        out[f"var_{k}_ci"] = [lo, hi]
        out[f"raw_{k}"] = [float(v) for v in arr]
    if entropy_cut is not None:
        out["entropy_mean"] = float(np.mean(ents))
        out["entropy_std"] = float(np.std(ents))
    return out


def sample_costs(build, ham, n_samples, rng, param_sampler=None):
    """Only the cost C(theta) over random parameters (forward passes)."""
    costs = []
    for _ in range(n_samples):
        c = build(rng)
        params = c.random_params(rng) if param_sampler is None else param_sampler(c, rng)
        costs.append(ham.expectation(c.run(params)))
    costs = np.asarray(costs)
    lo, hi = bootstrap_var_ci(costs)
    return {"n_samples": n_samples, "cost_mean": float(costs.mean()), "cost_var": float(costs.var()),
            "cost_var_ci": [lo, hi]}


# ----------------------------------------------------------------------- optimisation
def adam(cost_and_grad: Callable, x0, steps: int, lr: float = 0.05, beta1: float = 0.9, beta2: float = 0.999,
         eps: float = 1e-8, callback: Optional[Callable] = None):
    """Minimise; ``cost_and_grad(x) -> (cost, grad)``.  Returns (x, history of costs)."""
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    hist = []
    for t in range(1, steps + 1):
        c, g = cost_and_grad(x)
        hist.append(float(c))
        if callback is not None:
            callback(t, x, c, g)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mh = m / (1 - beta1**t)
        vh = v / (1 - beta2**t)
        x = x - lr * mh / (np.sqrt(vh) + eps)
    return x, hist


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------- time budget
# run_all.sh exports DEADLINE_TS (unix seconds); scripts stop starting new points after it.
DEADLINE = float(os.environ.get("DEADLINE_TS", "1e18"))


def out_of_time() -> bool:
    return time.time() > DEADLINE
