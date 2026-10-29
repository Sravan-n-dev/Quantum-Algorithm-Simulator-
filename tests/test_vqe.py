"""Tests for VQE module."""

import numpy as np
import pytest

from qsim.vqe import (
    ising_hamiltonian,
    hardware_efficient_ansatz,
    run_vqe,
    exact_ground_energy,
)


class TestIsingHamiltonian:
    """Tests for Ising Hamiltonian construction."""

    def test_n2_term_count(self) -> None:
        """n=2 Ising should have 1 ZZ + 2 X = 3 terms."""
        H = ising_hamiltonian(2, J=1.0, h=1.0)
        # SparsePauliOp may simplify, but should have 3 non-trivial terms
        labels = [str(p) for p in H.paulis]
        # Should have ZZ, XI, IX (or equivalent)
        assert len(labels) >= 3, f"Expected >= 3 terms, got {labels}"

    def test_hamiltonian_hermitian(self) -> None:
        """Ising Hamiltonian must be Hermitian."""
        for n in [2, 3, 4]:
            H = ising_hamiltonian(n, J=1.0, h=0.5)
            mat = H.to_matrix()
            assert np.allclose(mat, mat.conj().T, atol=1e-10), (
                f"Hamiltonian for n={n} is not Hermitian"
            )

    def test_known_eigenvalue_n2(self) -> None:
        """Known ground energy for n=2, J=1, h=0."""
        # With h=0: H = -J * Z0Z1, eigenvalues are -J, -J, J, J
        H = ising_hamiltonian(2, J=1.0, h=0.0)
        e0 = exact_ground_energy(H)
        assert abs(e0 - (-1.0)) < 1e-10, f"Expected -1.0, got {e0}"


class TestVQE:
    """Tests for VQE optimization."""

    def test_vqe_n2_converges(self) -> None:
        """VQE for n=2 Ising should converge within 1e-3 of exact."""
        H = ising_hamiltonian(2, J=1.0, h=1.0)
        ansatz = hardware_efficient_ansatz(2, layers=2)
        exact = exact_ground_energy(H)

        result = run_vqe(H, ansatz, optimizer="COBYLA", seed=42, maxiter=500)
        error = abs(result['energy'] - exact)

        assert error < 1e-3, (
            f"VQE energy {result['energy']:.6f} vs exact {exact:.6f}, "
            f"error {error:.6f} > 1e-3"
        )

    def test_vqe_energy_below_max(self) -> None:
        """VQE energy should be below the maximum eigenvalue."""
        H = ising_hamiltonian(3, J=1.0, h=0.5)
        ansatz = hardware_efficient_ansatz(3, layers=1)
        result = run_vqe(H, ansatz, optimizer="COBYLA", seed=42, maxiter=200)

        mat = H.to_matrix()
        eigenvalues = np.linalg.eigvalsh(mat)
        assert result['energy'] < eigenvalues[-1], (
            "VQE should find energy below max eigenvalue"
        )

    def test_ansatz_parameterization(self) -> None:
        """Ansatz should have the correct number of parameters."""
        n, layers = 3, 2
        ansatz = hardware_efficient_ansatz(n, layers)
        expected_params = layers * 2 * n + 2 * n
        assert ansatz.num_parameters == expected_params, (
            f"Expected {expected_params} params, got {ansatz.num_parameters}"
        )
