"""Micro-benchmarks for qsim.

Measures, as a function of the qubit count n:
  * time per single-qubit gate, per CNOT, per generic 2-qubit gate, per diagonal ZZ rotation
  * forward pass of a hardware-efficient ansatz (HEA) with L layers
  * full gradient via the adjoint method and via the parameter-shift rule
  * (optional) HEA forward pass on PennyLane's default.qubit / lightning.qubit

Usage:  python benchmarks/bench_gates.py [--max-n 18] [--layers 10] [--no-pennylane]
Writes benchmarks/results.csv and benchmarks/bench.png
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qsim import Circuit, PauliSum, Statevector, adjoint_gradient, gates, parameter_shift_gradient  # noqa: E402
from qsim.gates import random_unitary  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def timeit(fn, min_time=0.3, max_repeats=50):
    """Median wall-clock seconds per call."""
    fn()  # warm-up
    times = []
    total = 0.0
    while total < min_time and len(times) < max_repeats:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        total += dt
    return float(np.median(times))


def hea(n, layers):
    """RY-RZ rotations + CNOT ladder, fresh parameter per rotation."""
    c = Circuit(n)
    for _ in range(layers):
        for q in range(n):
            c.ry(q)
            c.rz(q)
        for q in range(n - 1):
            c.cnot(q, q + 1)
    return c


def pennylane_hea_forward(n, layers, params):
    import pennylane as qml

    rows = {}
    for name in ["default.qubit", "lightning.qubit"]:
        try:
            dev = qml.device(name, wires=n)
        except Exception:
            continue

        @qml.qnode(dev)
        def f(p):
            k = 0
            for _ in range(layers):
                for q in range(n):
                    qml.RY(p[k], wires=q)
                    k += 1
                    qml.RZ(p[k], wires=q)
                    k += 1
                for q in range(n - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.expval(qml.PauliZ(0))

        rows[name] = timeit(lambda: f(params))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=18)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--no-pennylane", action="store_true")
    ap.add_argument("--ps-max-n", type=int, default=10, help="largest n for the parameter-shift timing")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    use_pl = not args.no_pennylane
    if use_pl:
        try:
            import pennylane  # noqa: F401
        except ImportError:
            use_pl = False

    rows = []
    for n in range(4, args.max_n + 1, 2):
        psi = Statevector.random(n, rng)
        U1 = gates.ry(0.3)
        U2 = random_unitary(4, rng)
        dzz = np.diagonal(gates.rzz(0.7))
        row = {"n": n, "layers": args.layers}
        row["gate_1q"] = timeit(lambda: psi.apply_1q(U1, n // 2))
        row["gate_cnot"] = timeit(lambda: psi.apply_cnot(0, n - 1))
        row["gate_2q_generic"] = timeit(lambda: psi.apply_2q(U2, 1, n - 2))
        row["gate_zz_diag"] = timeit(lambda: psi.apply_diag_2q(dzz, 1, n - 2))

        c = hea(n, args.layers)
        params = c.random_params(rng)
        H = PauliSum.local_z(n)
        row["n_params"] = c.n_params
        row["hea_forward"] = timeit(lambda: c.run(params))
        row["hea_adjoint_grad"] = timeit(lambda: adjoint_gradient(c, params, H))
        row["hea_param_shift_grad"] = (
            timeit(lambda: parameter_shift_gradient(c, params, H), min_time=0.1, max_repeats=3)
            if n <= args.ps_max_n else float("nan"))
        if use_pl:
            for name, t in pennylane_hea_forward(n, args.layers, params).items():
                row[f"pl_{name}_forward"] = t
        rows.append(row)
        print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "n", k != "layers", k))
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, "machine.txt"), "w") as fh:
        fh.write(f"python {platform.python_version()}  numpy {np.__version__}  {platform.platform()}  "
                 f"cpu={platform.processor() or 'unknown'}  cores={os.cpu_count()}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for key, label in [("gate_1q", "1-qubit gate"), ("gate_cnot", "CNOT (fast path)"),
                       ("gate_2q_generic", "generic 2-qubit gate"), ("gate_zz_diag", "diagonal ZZ rotation")]:
        ax.semilogy(ns, [r[key] for r in rows], "o-", label=label)
    ax.set_xlabel("qubits n")
    ax.set_ylabel("seconds per gate")
    ax.set_title("Per-gate cost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.semilogy(ns, [r["hea_forward"] for r in rows], "o-", label="qsim forward")
    ax.semilogy(ns, [r["hea_adjoint_grad"] for r in rows], "s-", label="qsim adjoint gradient (all params)")
    ps = [r["hea_param_shift_grad"] for r in rows]
    ax.semilogy(ns, ps, "^-", label="qsim parameter-shift gradient")
    for key, label in [("pl_default.qubit_forward", "PennyLane default.qubit forward"),
                       ("pl_lightning.qubit_forward", "PennyLane lightning.qubit forward")]:
        if any(key in r for r in rows):
            ax.semilogy(ns, [r.get(key, np.nan) for r in rows], "x--", label=label)
    ax.set_xlabel("qubits n")
    ax.set_ylabel("seconds")
    ax.set_title(f"Hardware-efficient ansatz, L={args.layers} layers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "bench.png"), dpi=150)
    print("wrote", os.path.join(HERE, "results.csv"), "and bench.png")


if __name__ == "__main__":
    main()
