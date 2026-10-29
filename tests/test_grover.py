"""Tests for Grover's search algorithm."""

import math

import pytest
from qiskit.quantum_info import Statevector

from qsim.grover import (
    build_oracle,
    build_diffuser,
    grover_circuit,
    analytical_success_probability,
    optimal_iterations,
)
from qsim.utils import success_probability_from_sv


class TestGroverOracle:
    """Tests for the Grover oracle."""

    def test_oracle_flips_marked_state(self) -> None:
        """Oracle should flip the sign of the marked state only."""
        n = 3
        marked = ['101']
        oracle = build_oracle(n, marked)

        # Start in equal superposition
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        qc.h(range(n))
        sv_before = Statevector(qc)

        # Apply oracle
        qc.compose(oracle, inplace=True)
        sv_after = Statevector(qc)

        # Check that only the marked state had its sign flipped
        for bs, amp_before in sv_before.to_dict().items():
            amp_after = sv_after.to_dict().get(bs, 0)
            if bs == '101':
                assert abs(amp_after + amp_before) < 1e-10, "Marked state should be flipped"
            else:
                assert abs(amp_after - amp_before) < 1e-10, f"State {bs} should be unchanged"

    def test_oracle_multiple_marked(self) -> None:
        """Oracle should flip signs of all marked states."""
        n = 3
        marked = ['010', '110']
        oracle = build_oracle(n, marked)

        from qiskit import QuantumCircuit
        qc = QuantumCircuit(n)
        qc.h(range(n))
        sv_before = Statevector(qc)
        qc.compose(oracle, inplace=True)
        sv_after = Statevector(qc)

        for bs, amp_before in sv_before.to_dict().items():
            amp_after = sv_after.to_dict().get(bs, 0)
            if bs in marked:
                assert abs(amp_after + amp_before) < 1e-10
            else:
                assert abs(amp_after - amp_before) < 1e-10


class TestGroverSuccessProbability:
    """Test that statevector matches analytical formula."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_sv_matches_analytical(self, n: int) -> None:
        """Statevector success prob must match analytical formula to 1e-8."""
        marked = ['0' * (n - 1) + '1']  # Mark state |1⟩
        M = 1
        k_opt = optimal_iterations(n, M)

        for k in range(k_opt + 4):
            qc = grover_circuit(n, marked, iterations=k)
            sv = Statevector(qc)
            sv_prob = success_probability_from_sv(sv, marked)
            analytical_prob = analytical_success_probability(n, M, k)
            assert abs(sv_prob - analytical_prob) < 1e-8, (
                f"n={n}, k={k}: SV prob {sv_prob:.10f} != analytical {analytical_prob:.10f}"
            )

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_optimal_k_high_success(self, n: int) -> None:
        """At optimal k with M=1, success probability must be >= 0.94 for n>=3."""
        marked = ['0' * (n - 1) + '1']
        k_opt = optimal_iterations(n, 1)
        qc = grover_circuit(n, marked, iterations=k_opt)
        sv = Statevector(qc)
        prob = success_probability_from_sv(sv, marked)
        assert prob >= 0.94, f"n={n}: success prob {prob:.4f} < 0.94"

    def test_n2_optimal_success(self) -> None:
        """n=2, M=1 at optimal k should give probability == 1.0."""
        n = 2
        marked = ['01']
        k_opt = optimal_iterations(n, 1)
        qc = grover_circuit(n, marked, iterations=k_opt)
        sv = Statevector(qc)
        prob = success_probability_from_sv(sv, marked)
        assert abs(prob - 1.0) < 1e-8, f"n=2: prob should be 1.0, got {prob}"

    def test_multiple_marked_states(self) -> None:
        """Grover with M > 1 marked states should still work."""
        n = 4
        marked = ['0001', '0010']
        M = 2
        k_opt = optimal_iterations(n, M)
        qc = grover_circuit(n, marked, iterations=k_opt)
        sv = Statevector(qc)
        prob = success_probability_from_sv(sv, marked)
        analytical = analytical_success_probability(n, M, k_opt)
        assert abs(prob - analytical) < 1e-8
