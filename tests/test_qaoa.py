"""Tests for QAOA module."""

import networkx as nx
import pytest

from qsim.qaoa import brute_force_maxcut, qaoa_circuit, run_qaoa, maxcut_hamiltonian


class TestBruteForceMaxCut:
    """Tests for the exact Max-Cut solver."""

    def test_triangle_cut(self) -> None:
        """Triangle graph should have max cut = 2."""
        G = nx.Graph([(0, 1), (1, 2), (0, 2)])
        cut_val, _ = brute_force_maxcut(G)
        assert cut_val == 2, f"Triangle max cut should be 2, got {cut_val}"

    def test_4_cycle_cut(self) -> None:
        """4-cycle should have max cut = 4."""
        G = nx.cycle_graph(4)
        cut_val, _ = brute_force_maxcut(G)
        assert cut_val == 4, f"4-cycle max cut should be 4, got {cut_val}"

    def test_single_edge(self) -> None:
        """Single edge graph should have max cut = 1."""
        G = nx.Graph([(0, 1)])
        cut_val, _ = brute_force_maxcut(G)
        assert cut_val == 1


class TestMaxCutHamiltonian:
    """Tests for the Max-Cut Hamiltonian."""

    def test_hamiltonian_hermitian(self) -> None:
        """Max-Cut Hamiltonian must be Hermitian."""
        import numpy as np
        G = nx.cycle_graph(4)
        H = maxcut_hamiltonian(G)
        mat = H.to_matrix()
        assert np.allclose(mat, mat.conj().T, atol=1e-10)


class TestQAOA:
    """Tests for QAOA execution."""

    def test_qaoa_ratio_in_valid_range(self) -> None:
        """QAOA approximation ratio should be in [0, 1]."""
        G = nx.Graph([(0, 1), (1, 2), (0, 2)])
        result = run_qaoa(G, p=1, optimizer="COBYLA", seed=42, restarts=3,
                          maxiter=100)
        assert 0 <= result['approx_ratio'] <= 1.0 + 1e-6, (
            f"Approx ratio {result['approx_ratio']} out of valid range"
        )

    def test_qaoa_triangle_reasonable(self) -> None:
        """QAOA on triangle should achieve reasonable approximation."""
        G = nx.Graph([(0, 1), (1, 2), (0, 2)])
        result = run_qaoa(G, p=2, optimizer="COBYLA", seed=42, restarts=5,
                          maxiter=200)
        assert result['approx_ratio'] > 0.5, (
            f"Approx ratio {result['approx_ratio']:.4f} too low on triangle"
        )

    def test_qaoa_circuit_structure(self) -> None:
        """QAOA circuit should have correct number of qubits."""
        G = nx.Graph([(0, 1), (1, 2)])
        qc = qaoa_circuit(G, [0.5], [0.3])
        assert qc.num_qubits == 3
