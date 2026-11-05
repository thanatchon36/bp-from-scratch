"""Dynamical Lie algebras (DLAs) of parameterised circuits.

The DLA ``g`` of a circuit with generators {H_k} is the real Lie algebra spanned
by all nested commutators of {i H_k}.  Ragone et al. (2024) show that for deep
circuits (a 2-design over e^g) and a cost  C = Tr[U rho U^dag H]  with rho or H in
i g, the cost variance is  Var[C] = P_g(rho) P_g(H) / dim(g),  where
P_g(X) = ||X_g||_F^2 is the squared Frobenius norm of the projection of X onto g.

Operators are stored in the Hermitian Pauli basis.  A Pauli string is encoded by
two n-bit integers (x, z) with the Hermitian convention  P = i^{|x&z|} X^x Z^z
(so (1,1) is Y).  Products and commutators are then bit operations.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Tuple

import numpy as np

Pauli = Tuple[int, int]  # (x bits, z bits)
Operator = Dict[Pauli, float]  # Hermitian: real coefficients


# ----------------------------------------------------------------------- encoding helpers
def label_to_pauli(label: str) -> Pauli:
    """'XZIY' (leftmost = qubit 0) -> (x, z) with qubit 0 as the most significant bit."""
    x = z = 0
    for ch in label.upper():
        x <<= 1
        z <<= 1
        if ch in "XY":
            x |= 1
        if ch in "ZY":
            z |= 1
    return x, z


def pauli_to_label(p: Pauli, n: int) -> str:
    x, z = p
    out = []
    for i in range(n - 1, -1, -1):
        xi, zi = (x >> i) & 1, (z >> i) & 1
        out.append("IXZY"[xi + 2 * zi])
    return "".join(out)


def pauli_mult(a: Pauli, b: Pauli) -> Tuple[int, Pauli]:
    """P_a P_b = i^k P_c.  Returns (k mod 4, c)."""
    xa, za = a
    xb, zb = b
    xc, zc = xa ^ xb, za ^ zb
    k = bin(xa & za).count("1") + bin(xb & zb).count("1") + 2 * bin(za & xb).count("1") - bin(xc & zc).count("1")
    return k % 4, (xc, zc)


def anticommute(a: Pauli, b: Pauli) -> bool:
    return (bin(a[0] & b[1]).count("1") + bin(a[1] & b[0]).count("1")) % 2 == 1


def commutator_i(A: Operator, B: Operator) -> Operator:
    """i[A, B] for Hermitian A, B (the result is Hermitian with real coefficients)."""
    out: Operator = {}
    for pa, ca in A.items():
        for pb, cb in B.items():
            if not anticommute(pa, pb):
                continue
            k, pc = pauli_mult(pa, pb)
            # i [P_a, P_b] = 2 i * i^k P_c = 2 i^{k+1} P_c   with k+1 even here
            sign = 1.0 if (k + 1) % 4 == 0 else -1.0
            out[pc] = out.get(pc, 0.0) + 2.0 * sign * ca * cb
    return {p: c for p, c in out.items() if abs(c) > 1e-12}


def operator_from_terms(terms: Iterable[Tuple[float, str]]) -> Operator:
    out: Operator = {}
    for c, label in terms:
        p = label_to_pauli(label)
        out[p] = out.get(p, 0.0) + float(c)
    return out


# ----------------------------------------------------------------------- closures
def pauli_string_closure(generators: Iterable[Pauli], cap: int | None = None) -> set:
    """Lie closure when every generator is a single Pauli string: the DLA is spanned by
    Pauli strings, so the closure is a breadth-first search over strings."""
    basis = set(generators)
    queue = deque(basis)
    gens = list(set(generators))
    while queue:
        p = queue.popleft()
        for g in gens:
            if anticommute(p, g):
                _, c = pauli_mult(p, g)
                if c not in basis:
                    basis.add(c)
                    queue.append(c)
                    if cap is not None and len(basis) > cap:
                        return basis
    return basis


class LieClosure:
    """Generic Lie closure with an orthonormal basis (Gram-Schmidt in the Pauli basis).
    Suitable when dim(g) is at most a few thousand."""

    def __init__(self, generators: List[Operator], cap: int = 4000, tol: float = 1e-9):
        self.gens = generators
        self.tol = tol
        self.cap = cap
        self.keys: Dict[Pauli, int] = {}
        self.Q = np.zeros((0, 0))  # orthonormal rows
        self.capped = False
        self._run()

    def _vec(self, op: Operator) -> np.ndarray:
        for p in op:
            if p not in self.keys:
                self.keys[p] = len(self.keys)
                if self.Q.shape[1] < len(self.keys):
                    self.Q = np.pad(self.Q, ((0, 0), (0, max(16, self.Q.shape[1]))))
        v = np.zeros(self.Q.shape[1])
        for p, c in op.items():
            v[self.keys[p]] = c
        return v

    def _add(self, op: Operator) -> bool:
        if not op:
            return False
        v = self._vec(op)
        if self.Q.shape[0]:
            v = v - self.Q.T @ (self.Q @ v)
            v = v - self.Q.T @ (self.Q @ v)  # re-orthogonalise
        nrm = np.linalg.norm(v)
        if nrm < self.tol * max(1.0, np.sqrt(sum(c * c for c in op.values()))):
            return False
        self.Q = np.vstack([self.Q, v / nrm]) if self.Q.shape[0] else (v / nrm)[None, :]
        return True

    def _row_op(self, i: int) -> Operator:
        row = self.Q[i]
        return {p: row[j] for p, j in self.keys.items() if abs(row[j]) > 1e-12}

    def _run(self):
        for g in self.gens:
            self._add(g)
        frontier = list(range(self.Q.shape[0]))
        while frontier:
            new = []
            for i in frontier:
                bi = self._row_op(i)
                for g in self.gens:
                    if self._add(commutator_i(bi, g)):
                        new.append(self.Q.shape[0] - 1)
                        if self.Q.shape[0] >= self.cap:
                            self.capped = True
                            return
            frontier = new

    @property
    def dim(self) -> int:
        return int(self.Q.shape[0])

    def purity(self, op: Operator, n: int) -> float:
        """P_g(X) = ||X_g||_F^2 for X given in the Pauli basis (coefficients c_P, X = sum c_P P)."""
        v = np.zeros(self.Q.shape[1])
        for p, c in op.items():
            if p in self.keys:
                v[self.keys[p]] = c
        return float(2**n * np.sum((self.Q @ v) ** 2))  # ||P||_F^2 = 2^n


def product_state_operator(n: int, basis_state: str = "0") -> Operator:
    """rho = |s><s|^{(x)n} for s in {'0', '+'} expanded in Pauli strings: rho = 2^-n sum_P <P> P."""
    out: Operator = {}
    diag_letter = "Z" if basis_state == "0" else "X"
    for mask in range(2**n):
        label = "".join(diag_letter if (mask >> (n - 1 - i)) & 1 else "I" for i in range(n))
        out[label_to_pauli(label)] = 2.0**-n
    return out


def set_purity(strings: set, op: Operator, n: int) -> float:
    """P_g(X) when g is spanned by the Pauli strings in ``strings``."""
    return float(2**n * sum(c * c for p, c in op.items() if p in strings))


def dla_prediction(dim: int, purity_rho: float, purity_h: float) -> float:
    """Ragone et al. (2024): Var[C] = P_g(rho) P_g(H) / dim(g)."""
    return purity_rho * purity_h / dim
