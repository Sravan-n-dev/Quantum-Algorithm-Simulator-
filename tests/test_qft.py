"""Tests for QFT module."""

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

from qsim.qft import qft_circuit, dft_matrix


class TestQFTCircuit:
    """Tests for the QFT circuit construction."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_qft_matches_dft_matrix(self, n: int) -> None:
        """QFT unitary must match the DFT matrix up to global phase."""
        qc = qft_circuit(n)
        qft_op = Operator(qc)
        dft = dft_matrix(n)

        # Check equivalence up to global phase
        assert qft_op.equiv(Operator(dft), atol=1e-8), (
            f"QFT(n={n}) does not match DFT matrix up to global phase"
        )

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_qft_inverse_is_identity(self, n: int) -> None:
        """QFT followed by inverse QFT must be identity."""
        qc = qft_circuit(n)
        qc_inv = qft_circuit(n, inverse=True)
        combined = qc.compose(qc_inv)
        op = Operator(combined)

        identity = np.eye(2 ** n)
        assert np.allclose(np.abs(op.data), identity, atol=1e-8), (
            f"QFT · IQFT is not identity for n={n}"
        )

    def test_qft_no_swaps(self) -> None:
        """QFT without swaps should still be unitary."""
        qc = qft_circuit(3, do_swaps=False)
        op = Operator(qc)
        # Check unitarity
        product = op.data @ op.data.conj().T
        assert np.allclose(product, np.eye(8), atol=1e-10)


class TestDFTMatrix:
    """Tests for the DFT matrix construction."""

    def test_dft_n1(self) -> None:
        """1-qubit DFT should be H/sqrt(2) equivalent."""
        dft = dft_matrix(1)
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        # They should match up to global phase
        assert np.allclose(np.abs(dft), np.abs(expected), atol=1e-10)

    def test_dft_unitary(self) -> None:
        """DFT matrix must be unitary."""
        for n in range(1, 6):
            dft = dft_matrix(n)
            product = dft @ dft.conj().T
            assert np.allclose(product, np.eye(2 ** n), atol=1e-10), (
                f"DFT matrix for n={n} is not unitary"
            )
