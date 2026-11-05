"""Cost operators.

Every operator exposes

* ``expectation(state) -> float``   the cost  <psi|H|psi>
* ``apply(state) -> Statevector``   H|psi>  (needed by adjoint differentiation)
* ``to_matrix() -> ndarray``        dense matrix, for tests on small n
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence, Tuple

import numpy as np

from .gates import C, pauli_matrix
from .state import Statevector


class PauliSum:
    """H = sum_k c_k P_k with real coefficients c_k and Pauli strings P_k."""

    def __init__(self, terms: Iterable[Tuple[float, str]], n: int | None = None):
        terms = [(float(c), str(l).upper()) for c, l in terms]
        if not terms:
            raise ValueError("PauliSum needs at least one term")
        lengths = {len(l) for _, l in terms}
        if len(lengths) != 1:
            raise ValueError("all Pauli strings must have the same length")
        self.n = int(n) if n is not None else lengths.pop()
        for _, l in terms:
            if len(l) != self.n or set(l) - set("IXYZ"):
                raise ValueError(f"bad Pauli string {l!r}")
        self.terms = terms

    # ------------------------------------------------------------ algebra
    def simplify(self, tol: float = 1e-14) -> "PauliSum":
        acc: "OrderedDict[str, float]" = OrderedDict()
        for c, l in self.terms:
            acc[l] = acc.get(l, 0.0) + c
        terms = [(c, l) for l, c in acc.items() if abs(c) > tol]
        return PauliSum(terms or [(0.0, "I" * self.n)], n=self.n)

    def __add__(self, other: "PauliSum") -> "PauliSum":
        if other.n != self.n:
            raise ValueError("qubit counts differ")
        return PauliSum(self.terms + other.terms, n=self.n).simplify()

    def __mul__(self, scalar: float) -> "PauliSum":
        return PauliSum([(c * float(scalar), l) for c, l in self.terms], n=self.n)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return "PauliSum(" + " + ".join(f"{c:+g}*{l}" for c, l in self.terms) + ")"

    # ------------------------------------------------------------ evaluation
    def expectation(self, state: Statevector) -> float:
        total = 0.0
        for c, label in self.terms:
            if set(label) == {"I"}:
                total += c * np.real(state.inner(state))
                continue
            phi = state.copy().apply_pauli_string(label)
            total += c * np.real(state.inner(phi))
        return float(total)

    def apply(self, state: Statevector) -> Statevector:
        out = np.zeros_like(state.data)
        for c, label in self.terms:
            if set(label) == {"I"}:
                out += c * state.data
            else:
                out += c * state.copy().apply_pauli_string(label).data
        return Statevector(state.n, out)

    def to_matrix(self) -> np.ndarray:
        m = np.zeros((2**self.n, 2**self.n), dtype=C)
        for c, label in self.terms:
            m += c * pauli_matrix(label)
        return m

    def frobenius_norm_sq(self) -> float:
        """Tr(H^2) computed from the (orthogonal) Pauli expansion: 2^n * sum c_k^2."""
        s = self.simplify()
        return float(2**self.n * sum(c * c for c, _ in s.terms))

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_label(cls, label: str, coeff: float = 1.0) -> "PauliSum":
        return cls([(coeff, label)])

    @classmethod
    def single(cls, n: int, q: int, pauli: str = "Z", coeff: float = 1.0) -> "PauliSum":
        label = ["I"] * n
        label[q] = pauli
        return cls([(coeff, "".join(label))], n=n)

    @classmethod
    def global_z(cls, n: int) -> "PauliSum":
        """Z (x) Z (x) ... (x) Z — a global, traceless cost operator."""
        return cls([(1.0, "Z" * n)], n=n)

    @classmethod
    def local_z(cls, n: int, normalize: bool = True) -> "PauliSum":
        """(1/n) sum_i Z_i (or the plain sum if ``normalize=False``)."""
        w = 1.0 / n if normalize else 1.0
        return cls([(w, "I" * i + "Z" + "I" * (n - i - 1)) for i in range(n)], n=n)

    @classmethod
    def maxcut(cls, n: int, edges: Sequence[Tuple[int, int]], weights=None) -> "PauliSum":
        """Cut-size operator  C = sum_{(i,j)} w_ij (1 - Z_i Z_j) / 2  (to be *maximised*)."""
        if weights is None:
            weights = [1.0] * len(edges)
        terms = []
        const = 0.0
        for (i, j), w in zip(edges, weights):
            label = ["I"] * n
            label[i] = "Z"
            label[j] = "Z"
            terms.append((-0.5 * w, "".join(label)))
            const += 0.5 * w
        terms.append((const, "I" * n))
        return cls(terms, n=n)


class Projector0:
    """|0...0><0...0| — the 'global' cost operator used by Cerezo et al. (2021)."""

    def __init__(self, n: int):
        self.n = int(n)

    def expectation(self, state: Statevector) -> float:
        return float(abs(state.data[0]) ** 2)

    def apply(self, state: Statevector) -> Statevector:
        out = np.zeros_like(state.data)
        out[0] = state.data[0]
        return Statevector(state.n, out)

    def to_matrix(self) -> np.ndarray:
        m = np.zeros((2**self.n, 2**self.n), dtype=C)
        m[0, 0] = 1.0
        return m

    def frobenius_norm_sq(self) -> float:
        return 1.0

    def __repr__(self) -> str:
        return f"Projector0(n={self.n})"
