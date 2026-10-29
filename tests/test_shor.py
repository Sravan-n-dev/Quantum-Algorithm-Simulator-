"""Tests for Shor's factoring algorithm."""

import math

import pytest

from qsim.shor import (
    classical_order,
    extract_order,
    order_finding_circuit,
    shor_factor,
    _modular_permutation_matrix,
)


class TestModularExponentiation:
    """Tests for the modular exponentiation components."""

    def test_permutation_matrix_unitarity(self) -> None:
        """Permutation matrix must be unitary."""
        import numpy as np
        for a in [2, 4, 7, 8, 11, 13]:
            n_target = math.ceil(math.log2(15))
            U = _modular_permutation_matrix(a, 15, n_target)
            product = U @ U.conj().T
            assert np.allclose(product, np.eye(2 ** n_target), atol=1e-10), (
                f"U_{a} mod 15 is not unitary"
            )

    def test_permutation_correctness(self) -> None:
        """U_a|y⟩ = |a*y mod N⟩ for y < N."""
        import numpy as np
        a, N = 7, 15
        n_target = math.ceil(math.log2(N))
        U = _modular_permutation_matrix(a, N, n_target)
        for y in range(N):
            expected = (a * y) % N
            state_in = np.zeros(2 ** n_target)
            state_in[y] = 1.0
            state_out = U @ state_in
            assert abs(state_out[expected] - 1.0) < 1e-10, (
                f"|{y}⟩ should map to |{expected}⟩ but didn't"
            )


class TestClassicalOrder:
    """Tests for classical order computation."""

    @pytest.mark.parametrize("a,expected_r", [
        (2, 4), (4, 2), (7, 4), (8, 4), (11, 2), (13, 4),
    ])
    def test_order_n15(self, a: int, expected_r: int) -> None:
        """Classical order of a mod 15 should match known values."""
        r = classical_order(a, 15)
        assert r == expected_r, f"ord({a}, 15) = {r}, expected {expected_r}"


class TestOrderFinding:
    """Tests for the quantum order finding circuit."""

    @pytest.mark.parametrize("a,expected_r", [
        (2, 4), (4, 2), (7, 4), (11, 2), (13, 4),
    ])
    def test_order_finding_n15(self, a: int, expected_r: int) -> None:
        """Order finding circuit should find the correct order for N=15."""
        from qiskit_aer import AerSimulator
        from qiskit import transpile

        N = 15
        qc = order_finding_circuit(a, N)
        t = 2 * math.ceil(math.log2(N))

        sim = AerSimulator(method='statevector')
        transpiled = transpile(qc, sim)
        result = sim.run(transpiled, shots=2048, seed_simulator=42).result()
        counts = result.get_counts()

        # Check that at least one measurement yields the correct order
        found_correct = False
        for bitstring in counts:
            measured_val = int(bitstring, 2)
            r = extract_order(measured_val, t, N, a)
            if r == expected_r:
                found_correct = True
                break

        assert found_correct, (
            f"Failed to find order {expected_r} for a={a}, N=15. Counts: {counts}"
        )


class TestShorFactor:
    """Tests for the full Shor factoring pipeline."""

    def test_factor_15(self) -> None:
        """shor_factor(15) must return correct factors."""
        p, q, attempts = shor_factor(15, seed=42)
        assert p * q == 15, f"Factors {p}*{q} != 15"
        assert p > 1 and q > 1, f"Trivial factors: {p}, {q}"

    def test_factor_21(self) -> None:
        """shor_factor(21) must return correct factors."""
        p, q, attempts = shor_factor(21, seed=42)
        assert p * q == 21, f"Factors {p}*{q} != 21"
        assert p > 1 and q > 1, f"Trivial factors: {p}, {q}"
