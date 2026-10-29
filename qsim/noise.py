"""Noise models and noisy simulation utilities.

Built on qiskit_aer.noise. Supports depolarizing errors, readout errors,
and combined models. Provides utilities for running circuits under noise
and computing success probabilities.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


# Basis gates used for noisy simulation
NOISE_BASIS_GATES = ['u', 'rz', 'sx', 'x', 'cx']


def depolarizing_noise_model(p1: float, p2: float) -> NoiseModel:
    """Create a depolarizing noise model.

    Args:
        p1: 1-qubit depolarizing error probability on single-qubit gates.
        p2: 2-qubit depolarizing error probability on CX gates.

    Returns:
        A NoiseModel with depolarizing errors on basis gates.
    """
    model = NoiseModel()

    if p1 > 0:
        error_1q = depolarizing_error(p1, 1)
        for gate in ['u', 'rz', 'sx', 'x']:
            model.add_all_qubit_quantum_error(error_1q, gate)

    if p2 > 0:
        error_2q = depolarizing_error(p2, 2)
        model.add_all_qubit_quantum_error(error_2q, 'cx')

    return model


def readout_noise_model(p01: float, p10: float) -> NoiseModel:
    """Create a readout-error-only noise model.

    Args:
        p01: Probability of reading 1 when state is 0.
        p10: Probability of reading 0 when state is 1.

    Returns:
        A NoiseModel with asymmetric readout errors.
    """
    model = NoiseModel()

    if p01 > 0 or p10 > 0:
        readout_err = ReadoutError(
            [[1 - p01, p01],
             [p10, 1 - p10]]
        )
        model.add_all_qubit_readout_error(readout_err)

    return model


def combined_noise_model(
    p1: float, p2: float, p01: float, p10: float
) -> NoiseModel:
    """Create a noise model with both gate and readout errors.

    Args:
        p1: 1-qubit depolarizing probability.
        p2: 2-qubit depolarizing probability.
        p01: Readout 0→1 flip probability.
        p10: Readout 1→0 flip probability.

    Returns:
        A combined NoiseModel.
    """
    model = NoiseModel()

    if p1 > 0:
        error_1q = depolarizing_error(p1, 1)
        for gate in ['u', 'rz', 'sx', 'x']:
            model.add_all_qubit_quantum_error(error_1q, gate)

    if p2 > 0:
        error_2q = depolarizing_error(p2, 2)
        model.add_all_qubit_quantum_error(error_2q, 'cx')

    if p01 > 0 or p10 > 0:
        readout_err = ReadoutError(
            [[1 - p01, p01],
             [p10, 1 - p10]]
        )
        model.add_all_qubit_readout_error(readout_err)

    return model


def run_with_noise(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    shots: int = 4096,
    seed: int = 42,
) -> dict[str, int]:
    """Run a circuit on a noisy AerSimulator.

    The circuit is transpiled to the noise model's basis gates before simulation.

    Args:
        circuit: QuantumCircuit with measurements.
        noise_model: The noise model to apply.
        shots: Number of measurement shots.
        seed: Simulator random seed.

    Returns:
        Measurement counts dictionary.
    """
    simulator = AerSimulator(noise_model=noise_model, basis_gates=NOISE_BASIS_GATES)
    transpiled = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(transpiled, shots=shots, seed_simulator=seed).result()
    return result.get_counts()


def success_probability(
    counts: dict[str, int],
    target_bitstrings: list[str],
) -> float:
    """Compute the probability of measuring a target outcome.

    Args:
        counts: Measurement counts from a simulation.
        target_bitstrings: List of target bitstrings.

    Returns:
        Fraction of shots that produced a target outcome.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    successes = sum(counts.get(bs, 0) for bs in target_bitstrings)
    return successes / total


def binomial_std_error(p: float, n_shots: int) -> float:
    """Compute the standard error for a binomial proportion.

    Args:
        p: Estimated probability.
        n_shots: Number of shots.

    Returns:
        Standard error sqrt(p*(1-p)/n).
    """
    import math
    return math.sqrt(p * (1 - p) / n_shots) if n_shots > 0 else 0.0


def mitigate_readout_error(
    counts: dict[str, int],
    p01: float,
    p10: float,
    num_qubits: int,
) -> dict[str, float]:
    """Mitigate readout errors by inverting the single-qubit calibration matrix.

    Constructs M_1 = [[1 - p01, p10], [p01, 1 - p10]] and applies
    (M_1^{-1})^{⊗ n} to the empirical probability vector.

    Args:
        counts: Raw noisy measurement counts.
        p01: 0 -> 1 readout error probability.
        p10: 1 -> 0 readout error probability.
        num_qubits: Number of qubits measured.

    Returns:
        Dictionary mapping bitstrings to mitigated probabilities.
    """
    import numpy as np

    total = sum(counts.values())
    if total == 0:
        return {}

    # If no noise, return normalized counts
    if p01 == 0 and p10 == 0:
        return {bs: c / total for bs, c in counts.items()}

    # Single-qubit response matrix: M[measured, true]
    M1 = np.array([
        [1.0 - p01, p10],
        [p01, 1.0 - p10],
    ])
    try:
        M1_inv = np.linalg.inv(M1)
    except np.linalg.LinAlgError:
        return {bs: c / total for bs, c in counts.items()}

    # Full calibration matrix inverse: M_n_inv = kron(M1_inv, ... M1_inv)
    Mn_inv = M1_inv
    for _ in range(num_qubits - 1):
        Mn_inv = np.kron(Mn_inv, M1_inv)

    # Build empirical probability vector of size 2^num_qubits
    dim = 2 ** num_qubits
    p_meas = np.zeros(dim)
    for bs, cnt in counts.items():
        idx = int(bs, 2)
        if idx < dim:
            p_meas[idx] = cnt / total

    # Apply inverse
    p_mitigated = Mn_inv @ p_meas

    # Least squares projection / clip negative values and renormalize
    p_mitigated = np.clip(p_mitigated, 0.0, None)
    s = np.sum(p_mitigated)
    if s > 0:
        p_mitigated /= s

    # Convert back to bitstring dictionary
    res = {}
    for idx in range(dim):
        prob = float(p_mitigated[idx])
        if prob > 1e-6:
            bs = f"{idx:0{num_qubits}b}"
            res[bs] = prob

    return res
