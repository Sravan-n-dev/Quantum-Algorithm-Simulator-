"""QAOA (Quantum Approximate Optimization Algorithm) for Max-Cut.

Builds cost and mixer unitaries from RZZ and RX gates respectively.
Optimization via SciPy minimizers over (gamma, beta) parameters.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize


def maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """Build the Max-Cut cost Hamiltonian.

    H_C = Σ_{(i,j)∈E} w_{ij} * 0.5 * (I - Z_i Z_j)

    Maximising ⟨H_C⟩ is equivalent to maximising the cut value.

    Args:
        graph: A NetworkX graph. Edge weights are used if present
               (default weight = 1).

    Returns:
        SparsePauliOp representing the cost Hamiltonian.
    """
    n = graph.number_of_nodes()
    pauli_list: list[tuple[str, complex]] = []

    for i, j, data in graph.edges(data=True):
        w = data.get('weight', 1.0)
        # 0.5 * w * I
        pauli_list.append(('I' * n, 0.5 * w))
        # -0.5 * w * Z_i Z_j
        label = ['I'] * n
        label[i] = 'Z'
        label[j] = 'Z'
        pauli_list.append((''.join(label), -0.5 * w))

    return SparsePauliOp.from_list(pauli_list).simplify()


def qaoa_circuit(graph: nx.Graph, gammas: list[float], betas: list[float]) -> QuantumCircuit:
    """Build a QAOA circuit for Max-Cut.

    Args:
        graph: A NetworkX graph.
        gammas: Cost layer parameters (one per layer).
        betas: Mixer layer parameters (one per layer).

    Returns:
        A QuantumCircuit implementing QAOA with p = len(gammas) layers.
    """
    assert len(gammas) == len(betas), "gammas and betas must have the same length"
    n = graph.number_of_nodes()
    p = len(gammas)

    qc = QuantumCircuit(n, name=f"QAOA(p={p})")

    # Initial state: |+⟩^n
    qc.h(range(n))

    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        # Cost unitary: exp(-i * gamma * H_C)
        # For each edge (i,j): exp(-i * gamma * 0.5 * (I - Z_i Z_j))
        # = exp(-i*gamma*0.5) * exp(i*gamma*0.5 * Z_i Z_j)
        # The global phase doesn't matter, so just apply RZZ
        for i, j, data in graph.edges(data=True):
            w = data.get('weight', 1.0)
            # RZZ(theta)|q1,q2⟩ = exp(-i * theta/2 * Z_i Z_j)
            # We want exp(i * gamma * 0.5 * w * Z_i Z_j)
            # = exp(-i * (-gamma * w) * Z_i Z_j / 2) = RZZ(-gamma * w)
            qc.rzz(-gamma * w, i, j)

        # Mixer unitary: exp(-i * beta * Σ X_i)
        for i in range(n):
            qc.rx(2 * beta, i)

    return qc


def brute_force_maxcut(graph: nx.Graph) -> tuple[int, list[str]]:
    """Find the maximum cut value by exhaustive enumeration.

    Args:
        graph: A NetworkX graph (n ≤ 14 recommended).

    Returns:
        Tuple of (max_cut_value, list_of_optimal_bitstrings).
        Bitstrings are in Qiskit little-endian convention.
    """
    n = graph.number_of_nodes()
    best_val = -1
    best_strings: list[str] = []

    for bits in product([0, 1], repeat=n):
        cut_val = 0.0
        for i, j, data in graph.edges(data=True):
            w = data.get('weight', 1.0)
            if bits[i] != bits[j]:
                cut_val += w
        cut_val_int = int(cut_val) if cut_val == int(cut_val) else cut_val

        if cut_val > best_val:
            best_val = cut_val
            # Convert to Qiskit little-endian bitstring
            best_strings = [''.join(str(bits[i]) for i in reversed(range(n)))]
        elif abs(cut_val - best_val) < 1e-10:
            best_strings.append(''.join(str(bits[i]) for i in reversed(range(n))))

    return int(best_val) if best_val == int(best_val) else best_val, best_strings


def evaluate_cut(bitstring: str, graph: nx.Graph) -> float:
    """Evaluate the cut value for a given bitstring.

    Args:
        bitstring: In Qiskit little-endian convention.
        graph: A NetworkX graph.

    Returns:
        The cut value.
    """
    n = graph.number_of_nodes()
    # Convert from Qiskit little-endian to node assignments
    bits = [int(bitstring[n - 1 - i]) for i in range(n)]
    cut_val = 0.0
    for i, j, data in graph.edges(data=True):
        w = data.get('weight', 1.0)
        if bits[i] != bits[j]:
            cut_val += w
    return cut_val


def run_qaoa(
    graph: nx.Graph,
    p: int,
    optimizer: str = "COBYLA",
    seed: int = 42,
    restarts: int = 5,
    maxiter: int = 300,
) -> dict[str, Any]:
    """Run QAOA to approximately solve Max-Cut.

    Args:
        graph: A NetworkX graph.
        p: Number of QAOA layers.
        optimizer: SciPy optimizer name.
        seed: Base random seed.
        restarts: Number of random restarts.
        maxiter: Max iterations per restart.

    Returns:
        Dictionary with keys:
        - 'expected_cut': Best expected cut value found.
        - 'gammas': Optimal gamma parameters.
        - 'betas': Optimal beta parameters.
        - 'optimal_cut': Exact optimal cut value.
        - 'approx_ratio': expected_cut / optimal_cut.
        - 'history': Energy history of best run.
        - 'best_bitstring': Most probable bitstring from optimal circuit.
        - 'probabilities': Full probability distribution.
    """
    n = graph.number_of_nodes()
    H_cost = maxcut_hamiltonian(graph)
    optimal_cut, optimal_strings = brute_force_maxcut(graph)

    best_result: dict[str, Any] | None = None
    best_energy = -float('inf')

    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        x0 = rng.uniform(0, np.pi, 2 * p)

        history: list[float] = []

        def objective(params: np.ndarray) -> float:
            gammas = params[:p]
            betas = params[p:]
            qc = qaoa_circuit(graph, gammas.tolist(), betas.tolist())
            sv = Statevector(qc)
            exp_val = float(sv.expectation_value(H_cost).real)
            history.append(exp_val)
            return -exp_val  # Minimize negative expected cut

        result = minimize(
            objective, x0, method=optimizer,
            options={'maxiter': maxiter},
        )

        exp_cut = -result.fun
        if exp_cut > best_energy:
            best_energy = exp_cut
            gammas_opt = result.x[:p].tolist()
            betas_opt = result.x[p:].tolist()

            # Get probability distribution
            qc_opt = qaoa_circuit(graph, gammas_opt, betas_opt)
            sv_opt = Statevector(qc_opt)
            probs = sv_opt.probabilities_dict()

            # Find most probable bitstring
            best_bs = max(probs, key=probs.get)

            best_result = {
                'expected_cut': exp_cut,
                'gammas': gammas_opt,
                'betas': betas_opt,
                'optimal_cut': optimal_cut,
                'optimal_strings': optimal_strings,
                'approx_ratio': exp_cut / optimal_cut if optimal_cut > 0 else 1.0,
                'history': history.copy(),
                'best_bitstring': best_bs,
                'probabilities': probs,
                'optimal_prob': sum(probs.get(s, 0.0) for s in optimal_strings),
            }

    return best_result
