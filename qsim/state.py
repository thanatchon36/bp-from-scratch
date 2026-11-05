"""Dense state-vector simulation of n qubits.

The state is a flat, C-contiguous ``complex128`` array of length ``2**n``.
Basis index ``j = sum_i b_i * 2**(n-1-i)``: qubit 0 is the most significant bit,
so ``data.reshape((2,)*n)[b_0, b_1, ..., b_{n-1}]`` is the amplitude of
``|b_0 b_1 ... b_{n-1}>``.

Every gate method works **in place** and returns ``self`` so calls can be chained.
Gates are applied by reshaping the state into a 3-tensor ``(2**q, 2, rest)``
(one qubit) or a 5-tensor ``(2**lo, 2, mid, 2, rest)`` (two qubits) and acting on
the small axes; no ``2**n x 2**n`` matrix is ever built.
"""
from __future__ import annotations

import numpy as np

from .gates import C


class Statevector:
    __slots__ = ("n", "data")

    def __init__(self, n: int, data=None):
        self.n = int(n)
        if data is None:
            self.data = np.zeros(2**self.n, dtype=C)
            self.data[0] = 1.0
        else:
            arr = np.array(data, dtype=C, order="C").reshape(-1)
            if arr.size != 2**self.n:
                raise ValueError(f"expected {2**self.n} amplitudes, got {arr.size}")
            self.data = arr

    # ------------------------------------------------------------------ constructors
    @classmethod
    def zero(cls, n: int) -> "Statevector":
        return cls(n)

    @classmethod
    def from_array(cls, arr) -> "Statevector":
        arr = np.asarray(arr).reshape(-1)
        n = int(round(np.log2(arr.size)))
        if 2**n != arr.size:
            raise ValueError("length must be a power of two")
        return cls(n, arr)

    @classmethod
    def random(cls, n: int, rng: np.random.Generator) -> "Statevector":
        """Haar-random pure state (normalised complex Gaussian vector)."""
        v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        return cls(n, v / np.linalg.norm(v))

    def copy(self) -> "Statevector":
        return Statevector(self.n, self.data)

    # ------------------------------------------------------------------ views
    def _view3(self, q: int):
        return self.data.reshape(2**q, 2, 2 ** (self.n - q - 1))

    def _view5(self, lo: int, hi: int):
        return self.data.reshape(2**lo, 2, 2 ** (hi - lo - 1), 2, 2 ** (self.n - hi - 1))

    @staticmethod
    def _order(q1: int, q2: int):
        if q1 == q2:
            raise ValueError("two-qubit gate needs two distinct qubits")
        return (q1, q2) if q1 < q2 else (q2, q1)

    # ------------------------------------------------------------------ one-qubit gates
    def apply_1q(self, U, q: int) -> "Statevector":
        v = self._view3(q)
        a = v[:, 0, :]
        b = v[:, 1, :]
        na = U[0, 0] * a + U[0, 1] * b
        nb = U[1, 0] * a + U[1, 1] * b
        v[:, 0, :] = na
        v[:, 1, :] = nb
        return self

    def apply_diag_1q(self, d, q: int) -> "Statevector":
        v = self._view3(q)
        v[:, 0, :] *= d[0]
        v[:, 1, :] *= d[1]
        return self

    # ------------------------------------------------------------------ two-qubit gates
    def apply_2q(self, U, q1: int, q2: int) -> "Statevector":
        """General 4x4 gate written in the |q1 q2> basis (q1 = more significant)."""
        lo, hi = self._order(q1, q2)
        v = self._view5(lo, hi)
        # slices ordered by the gate's basis index 2*a + b  (a = q1 bit, b = q2 bit)
        if q1 < q2:  # axis 1 <-> q1, axis 3 <-> q2
            s = (v[:, 0, :, 0, :], v[:, 0, :, 1, :], v[:, 1, :, 0, :], v[:, 1, :, 1, :])
        else:  # axis 1 <-> q2, axis 3 <-> q1
            s = (v[:, 0, :, 0, :], v[:, 1, :, 0, :], v[:, 0, :, 1, :], v[:, 1, :, 1, :])
        new = [U[i, 0] * s[0] + U[i, 1] * s[1] + U[i, 2] * s[2] + U[i, 3] * s[3] for i in range(4)]
        for i in range(4):
            s[i][...] = new[i]
        return self

    def apply_diag_2q(self, d, q1: int, q2: int) -> "Statevector":
        """Diagonal 4x4 gate given by its 4 diagonal entries, ordered by (q1 bit, q2 bit)."""
        lo, hi = self._order(q1, q2)
        v = self._view5(lo, hi)
        if q1 < q2:
            v[:, 0, :, 0, :] *= d[0]
            v[:, 0, :, 1, :] *= d[1]
            v[:, 1, :, 0, :] *= d[2]
            v[:, 1, :, 1, :] *= d[3]
        else:
            v[:, 0, :, 0, :] *= d[0]
            v[:, 1, :, 0, :] *= d[1]
            v[:, 0, :, 1, :] *= d[2]
            v[:, 1, :, 1, :] *= d[3]
        return self

    def apply_cnot(self, control: int, target: int) -> "Statevector":
        lo, hi = self._order(control, target)
        v = self._view5(lo, hi)
        if control < target:
            sub = v[:, 1, :, :, :]  # (2**lo, mid, 2, rest): target is axis 2
            tmp = sub[:, :, 0, :].copy()
            sub[:, :, 0, :] = sub[:, :, 1, :]
            sub[:, :, 1, :] = tmp
        else:
            sub = v[:, :, :, 1, :]  # (2**lo, 2, mid, rest): target is axis 1
            tmp = sub[:, 0, :, :].copy()
            sub[:, 0, :, :] = sub[:, 1, :, :]
            sub[:, 1, :, :] = tmp
        return self

    def apply_cz(self, q1: int, q2: int) -> "Statevector":
        lo, hi = self._order(q1, q2)
        v = self._view5(lo, hi)
        v[:, 1, :, 1, :] *= -1
        return self

    # ------------------------------------------------------------------ generic dispatch
    def apply_matrix(self, U, qubits) -> "Statevector":
        qubits = tuple(qubits)
        if len(qubits) == 1:
            return self.apply_1q(U, qubits[0])
        if len(qubits) == 2:
            return self.apply_2q(U, qubits[0], qubits[1])
        return self._apply_kq(U, qubits)

    def apply_diag(self, d, qubits) -> "Statevector":
        qubits = tuple(qubits)
        if len(qubits) == 1:
            return self.apply_diag_1q(d, qubits[0])
        if len(qubits) == 2:
            return self.apply_diag_2q(d, qubits[0], qubits[1])
        return self._apply_kq(np.diag(d), qubits)

    def _apply_kq(self, U, qubits) -> "Statevector":
        """k-qubit gate via tensordot (slower, used for k > 2 e.g. Haar blocks)."""
        k = len(qubits)
        t = self.data.reshape((2,) * self.n)
        G = np.asarray(U, dtype=C).reshape((2,) * (2 * k))
        res = np.tensordot(G, t, axes=(list(range(k, 2 * k)), list(qubits)))
        res = np.moveaxis(res, list(range(k)), list(qubits))
        self.data = np.ascontiguousarray(res).reshape(-1)
        return self

    def apply_pauli_string(self, label: str) -> "Statevector":
        """Apply a Pauli string such as ``'XIZY'`` (character i acts on qubit i)."""
        for q, ch in enumerate(label.upper()):
            if ch == "I":
                continue
            v = self._view3(q)
            if ch == "Z":
                v[:, 1, :] *= -1
            elif ch == "X":
                tmp = v[:, 0, :].copy()
                v[:, 0, :] = v[:, 1, :]
                v[:, 1, :] = tmp
            elif ch == "Y":  # Y|0> = i|1>,  Y|1> = -i|0>
                tmp = v[:, 0, :].copy()
                v[:, 0, :] = -1j * v[:, 1, :]
                v[:, 1, :] = 1j * tmp
            else:
                raise ValueError(f"unknown Pauli letter {ch!r}")
        return self

    # ------------------------------------------------------------------ measurements
    def inner(self, other: "Statevector") -> complex:
        """<self|other>."""
        return complex(np.vdot(self.data, other.data))

    def norm(self) -> float:
        return float(np.linalg.norm(self.data))

    def normalize(self) -> "Statevector":
        self.data /= np.linalg.norm(self.data)
        return self

    def probabilities(self) -> np.ndarray:
        return np.abs(self.data) ** 2

    def amplitude(self, bits: str) -> complex:
        return complex(self.data[int(bits, 2)])

    def expectation(self, ham) -> float:
        return ham.expectation(self)

    def reduced_density_matrix(self, qubits) -> np.ndarray:
        """Reduced density matrix of the subsystem ``qubits`` (in ascending order)."""
        A = sorted(int(q) for q in qubits)
        B = [q for q in range(self.n) if q not in A]
        t = self.data.reshape((2,) * self.n).transpose(A + B)
        m = t.reshape(2 ** len(A), 2 ** len(B))
        return m @ m.conj().T

    def entanglement_entropy(self, qubits, base: float = np.e) -> float:
        """Von Neumann entropy of the reduced state on ``qubits`` (default: nats)."""
        A = sorted(int(q) for q in qubits)
        B = [q for q in range(self.n) if q not in A]
        t = self.data.reshape((2,) * self.n).transpose(A + B)
        s = np.linalg.svd(t.reshape(2 ** len(A), 2 ** len(B)), compute_uv=False)
        p = s**2
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log(p)) / np.log(base))

    def __repr__(self) -> str:
        return f"Statevector(n={self.n}, norm={self.norm():.6f})"
