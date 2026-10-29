"""Grover's search algorithm implementation.

Builds oracle, diffuser, and full Grover circuit from basic gates
(X, H, multi-controlled Z). Includes analytical success probability.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate


def build_oracle(n: int, marked_states: list[str]) -> QuantumCircuit:
    """Build a phase oracle that flips the sign of marked basis states.

    Constructs the oracle from X gates and multi-controlled Z gates,
    *not* from a diagonal unitary matrix.

    Args:
        n: Number of qubits.
        marked_states: List of bitstrings (e.g. ['101', '010']) to mark.
            Each bitstring has length n, in big-endian order (MSB first).

    Returns:
        A QuantumCircuit acting as the phase oracle.
    """
    qc = QuantumCircuit(n, name="Oracle")

    for target in marked_states:
        assert len(target) == n, f"Bitstring {target!r} must have length {n}"

        # Flip qubits where the target bit is '0'
        # Qiskit is little-endian: qubit 0 is the rightmost bit
        for i in range(n):
            # big-endian index: target[n-1-i] corresponds to qubit i
            if target[n - 1 - i] == '0':
                qc.x(i)

        # Multi-controlled Z: Z on last qubit controlled by all others
        if n == 1:
            qc.z(0)
        else:
            mcz = ZGate().control(n - 1, annotated=True)
            qc.append(mcz, list(range(n)))

        # Undo the X flips
        for i in range(n):
            if target[n - 1 - i] == '0':
                qc.x(i)

    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    """Build the Grover diffuser (inversion about the mean).

    Implements: 2|s⟩⟨s| - I where |s⟩ = H^n|0⟩.
    Decomposition: H^n · (2|0⟩⟨0| - I) · H^n
                 = H^n · X^n · MCZ · X^n · H^n

    Args:
        n: Number of qubits.

    Returns:
        A QuantumCircuit implementing the diffuser.
    """
    qc = QuantumCircuit(n, name="Diffuser")

    # Apply H to all qubits
    qc.h(range(n))

    # Apply X to all qubits
    qc.x(range(n))

    # Multi-controlled Z
    if n == 1:
        qc.z(0)
    else:
        mcz = ZGate().control(n - 1, annotated=True)
        qc.append(mcz, list(range(n)))

    # Undo X and H
    qc.x(range(n))
    qc.h(range(n))

    return qc


def optimal_iterations(n: int, M: int) -> int:
    """Compute the optimal number of Grover iterations.

    Args:
        n: Number of qubits.
        M: Number of marked states.

    Returns:
        Optimal number of iterations: floor(pi/4 * sqrt(N/M)).
    """
    N = 2 ** n
    return max(1, math.floor(math.pi / 4 * math.sqrt(N / M)))


def grover_circuit(
    n: int,
    marked_states: list[str],
    iterations: int | None = None,
) -> QuantumCircuit:
    """Build a complete Grover search circuit.

    Args:
        n: Number of qubits.
        marked_states: List of bitstrings to search for.
        iterations: Number of Grover iterations. If None, uses the optimal count.

    Returns:
        A QuantumCircuit implementing Grover's algorithm.
    """
    M = len(marked_states)
    if iterations is None:
        iterations = optimal_iterations(n, M)

    oracle = build_oracle(n, marked_states)
    diffuser = build_diffuser(n)

    qc = QuantumCircuit(n, name=f"Grover(n={n}, k={iterations})")

    # Initial superposition
    qc.h(range(n))

    # Apply Grover iterations
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    return qc


def analytical_success_probability(n: int, M: int, k: int) -> float:
    """Compute the analytical success probability after k Grover iterations.

    P(success) = sin²((2k+1)·θ) where θ = arcsin(√(M/N)).

    Args:
        n: Number of qubits.
        M: Number of marked states.
        k: Number of iterations.

    Returns:
        The analytical probability of measuring a marked state.
    """
    N = 2 ** n
    theta = math.asin(math.sqrt(M / N))
    return math.sin((2 * k + 1) * theta) ** 2
