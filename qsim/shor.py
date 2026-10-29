"""Small-scale Shor's factoring algorithm.

Educational pipeline for N = 15 and N = 21 using order finding via
quantum phase estimation. Uses matrix-based modular exponentiation
oracles (UnitaryGate from permutation matrices), which is the standard
approach for small-scale demonstrations.
"""

from __future__ import annotations

import fractions
import math
import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from qsim.qft import qft_circuit


def _modular_permutation_matrix(a: int, N: int, n_target: int) -> np.ndarray:
    """Build the permutation matrix for U_a: |y⟩ → |a·y mod N⟩.

    For y >= N, the gate acts as identity.

    Args:
        a: Base for modular multiplication.
        N: Modulus.
        n_target: Number of qubits in the target register.

    Returns:
        Unitary matrix of size (2^n_target, 2^n_target).
    """
    dim = 2 ** n_target
    U = np.zeros((dim, dim), dtype=complex)
    for y in range(dim):
        if y < N:
            target = (a * y) % N
        else:
            target = y  # identity for y >= N
        U[target, y] = 1.0
    return U


def modular_exponentiation_gate(a: int, power: int, N: int, n_target: int) -> UnitaryGate:
    """Build U_a^power as a UnitaryGate.

    Computes a^power mod N and builds the corresponding permutation.

    Args:
        a: Base for modular multiplication.
        power: Exponent (typically 2^j for phase estimation).
        N: Modulus.
        n_target: Number of qubits in the target register.

    Returns:
        A UnitaryGate implementing |y⟩ → |a^power · y mod N⟩.
    """
    a_pow = pow(a, power, N)
    U = _modular_permutation_matrix(a_pow, N, n_target)
    return UnitaryGate(U, label=f"U_{a}^{power}")


def order_finding_circuit(a: int, N: int) -> QuantumCircuit:
    """Build the quantum order-finding circuit using phase estimation.

    Uses t = 2*ceil(log2(N)) counting qubits and ceil(log2(N)) target qubits.

    Args:
        a: Base for order finding (must be coprime to N).
        N: Number to find the order of a modulo N.

    Returns:
        A QuantumCircuit with measurements on the counting register.
    """
    n_target = math.ceil(math.log2(N))
    t = 2 * n_target  # counting register size

    qc = QuantumCircuit(t + n_target, t, name=f"OrderFinding(a={a}, N={N})")

    # Initialize target register to |1⟩
    qc.x(t)  # qubit t is the LSB of target register

    # Hadamards on counting register
    qc.h(range(t))

    # Controlled modular exponentiation
    for j in range(t):
        power = 2 ** j
        cu = modular_exponentiation_gate(a, power, N, n_target)
        controlled_cu = cu.control(1)
        qc.append(controlled_cu, [j] + list(range(t, t + n_target)))

    # Inverse QFT on counting register
    iqft = qft_circuit(t, inverse=True, do_swaps=True)
    qc.compose(iqft, qubits=range(t), inplace=True)

    # Measure counting register
    qc.measure(range(t), range(t))

    return qc


def extract_order(measured_value: int, t: int, N: int, a: int) -> int | None:
    """Extract the order from a measured phase estimation value.

    Uses continued fractions to find s/r from the measured phase ≈ s/r.

    Args:
        measured_value: Integer measured from the counting register.
        t: Number of counting qubits.
        N: Modulus.
        a: Base.

    Returns:
        The order r if successfully extracted, else None.
    """
    if measured_value == 0:
        return None

    # Phase = measured_value / 2^t
    phase = fractions.Fraction(measured_value, 2 ** t)

    # Use continued fractions to find r
    frac = phase.limit_denominator(N)
    r = frac.denominator

    # Verify: a^r mod N == 1
    if pow(a, r, N) == 1:
        return r

    # Try multiples of r
    for mult in range(2, N // r + 2):
        candidate = r * mult
        if candidate > N:
            break
        if pow(a, candidate, N) == 1:
            return candidate

    return None


def classical_order(a: int, N: int) -> int:
    """Compute the order of a modulo N by brute force.

    Args:
        a: Base.
        N: Modulus.

    Returns:
        The smallest positive r such that a^r ≡ 1 (mod N).
    """
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:
            raise ValueError(f"Order of {a} mod {N} exceeds {N}")
    return r


def _is_perfect_power(N: int) -> bool:
    """Check if N is a perfect power (N = a^b for b >= 2)."""
    for b in range(2, int(math.log2(N)) + 1):
        a = round(N ** (1.0 / b))
        for candidate in [a - 1, a, a + 1]:
            if candidate >= 2 and candidate ** b == N:
                return True
    return False


def shor_factor(N: int, seed: int = 42, max_attempts: int = 20) -> tuple[int, int, int]:
    """Factor N using Shor's algorithm.

    Classical wrapper that handles:
    - Even N, perfect powers, trivial gcd cases
    - Random base selection, order finding, factor extraction

    Args:
        N: Number to factor (must be composite, > 1).
        seed: Random seed for reproducibility.
        max_attempts: Maximum number of attempts before giving up.

    Returns:
        Tuple (p, q, attempts) where p*q = N and attempts is the number tried.

    Raises:
        ValueError: If factoring fails after max_attempts.
    """
    rng = random.Random(seed)

    # Handle trivial cases
    if N % 2 == 0:
        return 2, N // 2, 1
    if _is_perfect_power(N):
        for b in range(2, int(math.log2(N)) + 1):
            a = round(N ** (1.0 / b))
            for candidate in [a - 1, a, a + 1]:
                if candidate >= 2 and candidate ** b == N:
                    return candidate, N // candidate, 1

    from qiskit_aer import AerSimulator
    from qiskit import transpile

    simulator = AerSimulator(method='statevector')

    for attempt in range(1, max_attempts + 1):
        # Pick random a coprime to N
        a = rng.randint(2, N - 1)
        g = math.gcd(a, N)
        if g > 1:
            return g, N // g, attempt

        # Build and run order-finding circuit
        qc = order_finding_circuit(a, N)
        t = 2 * math.ceil(math.log2(N))
        transpiled = transpile(qc, simulator)
        result = simulator.run(transpiled, shots=2048, seed_simulator=seed + attempt).result()
        counts = result.get_counts()

        # Try each measurement outcome
        for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
            measured_val = int(bitstring, 2)
            r = extract_order(measured_val, t, N, a)
            if r is None or r == 0:
                continue
            if r % 2 != 0:
                continue

            # Try to extract factors
            x = pow(a, r // 2, N)
            if x == N - 1:  # a^(r/2) ≡ -1 (mod N)
                continue

            p = math.gcd(x + 1, N)
            q = math.gcd(x - 1, N)

            if p != 1 and p != N and N % p == 0:
                return p, N // p, attempt
            if q != 1 and q != N and N % q == 0:
                return q, N // q, attempt

    raise ValueError(f"Failed to factor {N} after {max_attempts} attempts")
