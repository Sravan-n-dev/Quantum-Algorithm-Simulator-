"""Tests for noise module."""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qsim.grover import grover_circuit, optimal_iterations
from qsim.noise import (
    depolarizing_noise_model,
    readout_noise_model,
    combined_noise_model,
    run_with_noise,
    success_probability,
    mitigate_readout_error,
)


class TestNoiseModels:
    """Tests for noise model construction."""

    def test_zero_noise_equals_ideal(self) -> None:
        """Zero error rates should give results matching ideal simulation."""
        n = 3
        marked = ['101']
        k_opt = optimal_iterations(n, 1)
        qc = grover_circuit(n, marked, iterations=k_opt)
        qc.measure_all()

        # Run with zero noise
        model = depolarizing_noise_model(0.0, 0.0)
        counts = run_with_noise(qc, model, shots=8192, seed=42)
        prob = success_probability(counts, marked)

        # Ideal probability from statevector
        qc_no_meas = grover_circuit(n, marked, iterations=k_opt)
        sv = Statevector(qc_no_meas)
        ideal_prob = sum(
            sv.probabilities_dict().get(bs, 0.0) for bs in marked
        )

        # Should match within shot noise (3 sigma ≈ 0.02 for 8192 shots)
        assert abs(prob - ideal_prob) < 0.03, (
            f"Zero-noise prob {prob:.4f} differs from ideal {ideal_prob:.4f}"
        )

    def test_success_decreases_with_noise(self) -> None:
        """Success probability should generally decrease as noise increases."""
        n = 3
        marked = ['101']
        k_opt = optimal_iterations(n, 1)
        qc = grover_circuit(n, marked, iterations=k_opt)
        qc.measure_all()

        probs = []
        for p2 in [0.0, 0.01, 0.03, 0.05]:
            model = depolarizing_noise_model(0.0, p2)
            counts = run_with_noise(qc, model, shots=4096, seed=42)
            prob = success_probability(counts, marked)
            probs.append(prob)

        # Overall trend should be decreasing (allow some noise)
        assert probs[0] > probs[-1], (
            f"Success should decrease with noise: {probs}"
        )

    def test_readout_error_only(self) -> None:
        """Readout errors should reduce measured success probability."""
        n = 3
        marked = ['101']
        k_opt = optimal_iterations(n, 1)
        qc = grover_circuit(n, marked, iterations=k_opt)
        qc.measure_all()

        # Run with readout error only
        model = readout_noise_model(0.05, 0.05)
        counts = run_with_noise(qc, model, shots=4096, seed=42)
        prob = success_probability(counts, marked)

        # Should be less than ideal but still reasonable
        assert prob > 0.5, f"Readout-only prob {prob:.4f} unexpectedly low"
        assert prob < 1.0, "Should not be exactly 1.0 with readout error"

    def test_combined_model(self) -> None:
        """Combined model should create a valid noise model."""
        model = combined_noise_model(0.001, 0.01, 0.02, 0.03)
        # Just check it runs without error
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        counts = run_with_noise(qc, model, shots=1024, seed=42)
        assert sum(counts.values()) == 1024

    def test_readout_error_mitigation(self) -> None:
        """Readout error mitigation should improve fidelity of recovered state."""
        n = 2
        qc = QuantumCircuit(n)
        qc.x(0)  # Prepare |01> (little-endian: q1=0, q0=1 -> '01')
        qc.measure_all()

        p01, p10 = 0.08, 0.08
        model = readout_noise_model(p01, p10)
        counts = run_with_noise(qc, model, shots=8192, seed=42)

        raw_prob = counts.get('01', 0) / 8192
        mitigated_probs = mitigate_readout_error(counts, p01, p10, num_qubits=n)
        mitigated_prob = mitigated_probs.get('01', 0.0)

        assert raw_prob < 0.90, f"Raw probability {raw_prob:.4f} unexpectedly high"
        assert mitigated_prob > raw_prob, (
            f"Mitigated prob {mitigated_prob:.4f} should exceed raw {raw_prob:.4f}"
        )
        assert mitigated_prob > 0.95, f"Mitigated prob {mitigated_prob:.4f} should be > 0.95"
