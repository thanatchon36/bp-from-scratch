"""Layered ansätze used in the experiments.

All builders return a :class:`qsim.Circuit`; parameters are created in circuit
order.  Qubits form an open 1D chain; CZ entanglers are applied either as a
*ladder* (all bonds (q, q+1) each layer) or as a *brick* pattern (bonds starting
at ``layer % 2``, i.e. the alternating-layered ansatz of Cerezo et al.).
"""
from __future__ import annotations

import numpy as np

from qsim import Circuit


def cz_layer(c: Circuit, n: int, layer: int = 0, brick: bool = False) -> None:
    start = layer % 2 if brick else 0
    for q in range(start, n - 1, 2 if brick else 1):
        c.cz(q, q + 1)


def hea_mcclean(n: int, L: int, rng: np.random.Generator, init_layer: bool = True) -> Circuit:
    """Random circuit of McClean et al. (2018): a fixed RY(pi/4) layer, then L layers of
    single-qubit rotations about an axis drawn uniformly from {X, Y, Z} per gate, each
    followed by a CZ ladder.  n*L parameters."""
    c = Circuit(n)
    if init_layer:
        for q in range(n):
            c.ry(q, angle=np.pi / 4)
    axes = rng.integers(0, 3, size=(L, n))
    for layer in range(L):
        for q in range(n):
            (c.rx, c.ry, c.rz)[axes[layer, q]](q)
        cz_layer(c, n, layer)
    c.axes = axes  # type: ignore[attr-defined]
    return c


def probe_indices(n: int, L: int) -> dict:
    """Parameter indices of the first, middle and last rotation of :func:`hea_mcclean`."""
    return {"first": 0, "mid": (L // 2) * n + n // 2, "last": L * n - 1}


def hea_ryrz(n: int, L: int, brick: bool = False) -> Circuit:
    """L layers of [RY, RZ on every qubit] + CZ entanglers (ladder or brick).  2nL parameters."""
    c = Circuit(n)
    for layer in range(L):
        for q in range(n):
            c.ry(q)
            c.rz(q)
        cz_layer(c, n, layer, brick=brick)
    return c


def ala(n: int, L: int) -> Circuit:
    """Alternating-layered ansatz: RY-RZ rotations and CZ gates in a brick pattern."""
    return hea_ryrz(n, L, brick=True)


def identity_block_hea(n: int, M: int, rng: np.random.Generator):
    """Identity-block initialisation of Grant et al. (2019).

    Each of the M blocks is  B(theta') A(theta)  with  A = [RY RZ] CZ [RY RZ]  and
    B = [RZ RY] CZ [RZ RY]  with its own parameters, initialised to theta' = -theta
    (in reverse gate order) so that every block is the identity at initialisation.
    Returns (circuit, init_params).  2M CZ layers, 8nM parameters.
    """
    c = Circuit(n)
    init = []
    for _ in range(M):
        a = rng.uniform(0, 2 * np.pi, size=(4, n))  # a[0]=RY, a[1]=RZ, a[2]=RY, a[3]=RZ  (first half)
        for q in range(n):  # A: [RY RZ]
            c.ry(q)
            c.rz(q)
            init += [a[0, q], a[1, q]]
        cz_layer(c, n)
        for q in range(n):  # A: [RY RZ]
            c.ry(q)
            c.rz(q)
            init += [a[2, q], a[3, q]]
        for q in range(n):  # B = A^dagger: [RZ RY] with negated angles, reversed order
            c.rz(q)
            c.ry(q)
            init += [-a[3, q], -a[2, q]]
        cz_layer(c, n)
        for q in range(n):
            c.rz(q)
            c.ry(q)
            init += [-a[1, q], -a[0, q]]
    return c, np.array(init)


def pauli_hea(n: int, L: int, full: bool = False) -> Circuit:
    """Fully parameterised Pauli-rotation HEA: per layer RY (and RX if ``full``) on every
    qubit and RZZ on every bond, each with its own parameter.  Generators
    {Y_i, Z_i Z_{i+1}}  or  {X_i, Y_i, Z_i Z_{i+1}}."""
    c = Circuit(n)
    for _ in range(L):
        for q in range(n):
            if full:
                c.rx(q)
            c.ry(q)
        for q in range(n - 1):
            c.rzz(q, q + 1)
    return c
