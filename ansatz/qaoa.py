"""Problem-inspired (Hamiltonian variational) ansätze."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from qsim import Circuit, PauliSum


def random_regular_graph(n: int, degree: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    import networkx as nx

    g = nx.random_regular_graph(degree, n, seed=int(rng.integers(0, 2**31 - 1)))
    return [(int(a), int(b)) for a, b in sorted(g.edges())]


def maxcut_value(n: int, edges) -> int:
    """Exact maximum cut by enumeration (n <= 20)."""
    idx = np.arange(2**n)
    bits = (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    cut = np.zeros(2**n, dtype=int)
    for i, j in edges:
        cut += bits[:, i] != bits[:, j]
    return int(cut.max())


def qaoa_maxcut(n: int, edges, p: int) -> Circuit:
    """QAOA: |+>^n then p layers of  exp(-i gamma sum_E Z_i Z_j)  exp(-i beta sum_i X_i).
    Parameters are ordered (gamma_1, beta_1, ..., gamma_p, beta_p).  The cost operator is
    ``PauliSum.maxcut(n, edges)`` (cut size, to be maximised)."""
    c = Circuit(n)
    for q in range(n):
        c.h(q)
    for layer in range(p):
        g, b = 2 * layer, 2 * layer + 1
        for i, j in edges:
            c.rzz(i, j, param=g, scale=2.0)
        for q in range(n):
            c.rx(q, param=b, scale=2.0)
    return c


def hva_tfim(n: int, L: int, plus_start: bool = False) -> Circuit:
    """Hamiltonian variational ansatz for the transverse-field Ising chain:
    L layers of  exp(-i phi sum_i Z_i Z_{i+1})  exp(-i theta sum_i X_i)  (2L parameters)."""
    c = Circuit(n)
    if plus_start:
        for q in range(n):
            c.h(q)
    for layer in range(L):
        a, b = 2 * layer, 2 * layer + 1
        for q in range(n - 1):
            c.rzz(q, q + 1, param=a, scale=2.0)
        for q in range(n):
            c.rx(q, param=b, scale=2.0)
    return c


def zz_chain(n: int) -> PauliSum:
    """H = sum_i Z_i Z_{i+1} on the open chain."""
    return PauliSum([(1.0, "I" * i + "ZZ" + "I" * (n - i - 2)) for i in range(n - 1)], n=n)
