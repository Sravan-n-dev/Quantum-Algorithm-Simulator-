"""Quantum Fourier Transform (QFT) implementation.

Hand-built from H, controlled-phase (CPhaseGate) and final swaps.
Does NOT use qiskit.circuit.library.QFT.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CPhaseGate


def qft_circuit(n: int, inverse: bool = False, do_swaps: bool = True) -> QuantumCircuit:
    """Build an n-qubit Quantum Fourier Transform circuit.

    The circuit applies H gates and controlled-phase rotations following
    the standard QFT decomposition (Nielsen & Chuang, Ch. 5).

    Args:
        n: Number of qubits.
        inverse: If True, build the inverse QFT (hermitian conjugate).
        do_swaps: If True, include the final SWAP gates to reverse qubit order.

    Returns:
        A QuantumCircuit implementing the (inverse) QFT.
    """
    qc = QuantumCircuit(n, name="QFT" if not inverse else "IQFT")

    if not inverse:
        # Forward QFT
        for i in range(n):
            qc.h(i)
            for j in range(i + 1, n):
                k = j - i + 1
                angle = 2 * np.pi / (2 ** k)
                qc.append(CPhaseGate(angle), [j, i])
        if do_swaps:
            for i in range(n // 2):
                qc.swap(i, n - 1 - i)
    else:
        # Inverse QFT: reverse order of operations + negate phases
        if do_swaps:
            for i in range(n // 2):
                qc.swap(i, n - 1 - i)
        for i in reversed(range(n)):
            for j in reversed(range(i + 1, n)):
                k = j - i + 1
                angle = -2 * np.pi / (2 ** k)
                qc.append(CPhaseGate(angle), [j, i])
            qc.h(i)

    return qc


def dft_matrix(n: int) -> np.ndarray:
    """Build the DFT matrix for n qubits, matching Qiskit's little-endian ordering.

    F[j, k] = exp(2πi·jk / N) / sqrt(N)  where N = 2^n.

    Qiskit uses little-endian qubit ordering: qubit 0 is the *least*
    significant bit. The standard QFT maps |j⟩ → Σ_k exp(2πi jk/N)/√N |k⟩
    in the computational basis where qubit order is reversed relative to
    the big-endian convention.

    To match Qiskit's convention, we apply bit-reversal permutation to
    both rows and columns of the standard DFT matrix.

    Args:
        n: Number of qubits.

    Returns:
        Complex ndarray of shape (2^n, 2^n).
    """
    N = 2 ** n
    # Standard DFT matrix
    j_indices = np.arange(N)
    k_indices = np.arange(N)
    F = np.exp(2j * np.pi * np.outer(j_indices, k_indices) / N) / np.sqrt(N)

    # Bit-reversal permutation for Qiskit's little-endian convention
    def reverse_bits(x: int, nbits: int) -> int:
        result = 0
        for _ in range(nbits):
            result = (result << 1) | (x & 1)
            x >>= 1
        return result

    perm = [reverse_bits(i, n) for i in range(N)]
    # Apply permutation to both rows and columns
    F = F[np.ix_(perm, perm)]

    return F
