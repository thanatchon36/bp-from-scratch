"""E0 — numerical check of the closed-form Haar-average gradient variances (appendix).

For psi = U_L R(theta) U_R |0>, dC/dtheta = 2 Re <psi| H U_L (-i/2) G R U_R |0>.
We sample U_L, U_R Haar (both, left-only or right-only) with G = Y on qubit 0 and compare
Var[dC/dtheta] with the formulas in experiments/common.py.
"""
import argparse

import numpy as np

from common import (haar_variance_left_only, haar_variance_right_only, haar_variance_two_sided, load, log, save)
from qsim import Projector0, PauliSum, Statevector, gates
from qsim.gates import random_unitary


def derivative(UL, UR, theta, n, ham):
    d = 2**n
    e0 = np.zeros(d, complex)
    e0[0] = 1.0
    G = np.kron(gates.Y, np.eye(d // 2))
    R = gates.rotation(G, theta)
    pre = R @ (UR @ e0 if UR is not None else e0)
    psi = UL @ pre if UL is not None else pre
    dpsi = (-0.5j) * (G @ pre)
    dpsi = UL @ dpsi if UL is not None else dpsi
    Hpsi = ham.apply(Statevector(n, psi)).data
    return 2.0 * np.real(np.vdot(Hpsi, dpsi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--samples", type=int, default=400)
    args = ap.parse_args()
    for n in args.n_list:
        key = f"n{n:02d}"
        if load("e0", key):
            continue
        rng = np.random.default_rng(100 + n)
        d = 2**n
        out = {"n": n, "samples": args.samples}
        hams = {"globalZ": (PauliSum.global_z(n), 0.0, float(d)), "projector": (Projector0(n), 1.0, 1.0)}
        for hname, (ham, trH, trH2) in hams.items():
            grads = {"two_sided": [], "left_only": [], "right_only": []}
            for _ in range(args.samples):
                UL, UR = random_unitary(d, rng), random_unitary(d, rng)
                th = rng.uniform(0, 2 * np.pi)
                grads["two_sided"].append(derivative(UL, UR, th, n, ham))
                grads["left_only"].append(derivative(UL, None, th, n, ham))
                grads["right_only"].append(derivative(None, UR, th, n, ham))
            # right-only prediction needs Tr(G H G H) with G = Y_0:  Y Z Y Z = -Z^2 -> -d for Z^n;
            # for the projector: Tr(G P G P) = |<0|G|0>|^2 = 0.
            trGHGH = -float(d) if hname == "globalZ" else 0.0
            out[hname] = {
                "measured": {k: float(np.var(v)) for k, v in grads.items()},
                "predicted": {
                    "two_sided": haar_variance_two_sided(d, trH, trH2),
                    "left_only": haar_variance_left_only(d, trH, trH2, trK2=2.0),  # <Y>_0 = 0
                    "right_only": haar_variance_right_only(d, trH2, trGHGH),
                },
            }
        save("e0", key, out)
        log(f"e0 n={n}: " + "  ".join(
            f"{h}:{k}={out[h]['measured'][k]:.3e}/{out[h]['predicted'][k]:.3e}"
            for h in hams for k in ["two_sided", "left_only", "right_only"]))


if __name__ == "__main__":
    main()
