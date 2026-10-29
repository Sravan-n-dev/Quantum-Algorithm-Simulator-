"""VQE (Variational Quantum Eigensolver) for Ising Hamiltonians.

Implements a hardware-efficient ansatz with RY/RZ rotations and CX entangling,
optimized via SciPy minimizers. Energy evaluation uses Statevector expectation values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize


def ising_hamiltonian(
    n: int,
    J: float = 1.0,
    h: float = 1.0,
    periodic: bool = False,
) -> SparsePauliOp:
    """Build the transverse-field Ising model Hamiltonian.

    H = -J Σ Z_i Z_{i+1} - h Σ X_i

    Args:
        n: Number of qubits (spins).
        J: ZZ coupling strength.
        h: Transverse field strength.
        periodic: If True, add Z_{n-1} Z_0 coupling.

    Returns:
        SparsePauliOp representing the Hamiltonian.
    """
    pauli_list: list[tuple[str, complex]] = []

    # ZZ interactions: -J * Z_i Z_{i+1}
    pairs = list(range(n - 1))
    if periodic and n > 2:
        pairs.append(n - 1)  # wraps around

    for i in pairs:
        j = (i + 1) % n
        # Build Pauli string: Z on qubit i and j, I elsewhere
        label = ['I'] * n
        label[i] = 'Z'
        label[j] = 'Z'
        pauli_list.append((''.join(label), -J))

    # Transverse field: -h * X_i
    for i in range(n):
        label = ['I'] * n
        label[i] = 'X'
        pauli_list.append((''.join(label), -h))

    return SparsePauliOp.from_list(pauli_list).simplify()


def hardware_efficient_ansatz(n: int, layers: int) -> QuantumCircuit:
    """Build a hardware-efficient variational ansatz.

    Each layer consists of:
    - RY rotation on each qubit
    - RZ rotation on each qubit
    - CX entangling ladder (linear connectivity)

    Args:
        n: Number of qubits.
        layers: Number of variational layers.

    Returns:
        Parameterized QuantumCircuit.
    """
    num_params = layers * 2 * n + 2 * n  # layers * (RY + RZ) per qubit + final RY + RZ
    params = ParameterVector('θ', num_params)

    qc = QuantumCircuit(n, name=f"HEA(n={n}, L={layers})")
    idx = 0

    for layer in range(layers):
        # RY rotations
        for q in range(n):
            qc.ry(params[idx], q)
            idx += 1
        # RZ rotations
        for q in range(n):
            qc.rz(params[idx], q)
            idx += 1
        # CX entangling ladder
        for q in range(n - 1):
            qc.cx(q, q + 1)

    # Final rotation layer (no entangling)
    for q in range(n):
        qc.ry(params[idx], q)
        idx += 1
    for q in range(n):
        qc.rz(params[idx], q)
        idx += 1

    return qc


def exact_ground_energy(H: SparsePauliOp) -> float:
    """Compute the exact ground state energy classically.

    Args:
        H: Hamiltonian as SparsePauliOp.

    Returns:
        The minimum eigenvalue of H.
    """
    matrix = H.to_matrix()
    eigenvalues = np.linalg.eigvalsh(matrix.real if np.allclose(matrix.imag, 0) else matrix)
    return float(eigenvalues[0])


def run_vqe(
    H: SparsePauliOp,
    ansatz: QuantumCircuit,
    optimizer: str = "COBYLA",
    seed: int = 42,
    maxiter: int = 500,
) -> dict[str, Any]:
    """Run VQE to find the ground state energy of H.

    Uses Statevector simulation for exact expectation values.

    Args:
        H: Hamiltonian as SparsePauliOp.
        ansatz: Parameterized quantum circuit.
        optimizer: SciPy optimizer name ('COBYLA', 'L-BFGS-B', etc.).
        seed: Random seed for initial parameters.
        maxiter: Maximum optimizer iterations.

    Returns:
        Dictionary with keys:
        - 'energy': Final optimized energy.
        - 'params': Optimal parameter values.
        - 'history': List of energies at each iteration.
        - 'num_evals': Number of function evaluations.
    """
    rng = np.random.default_rng(seed)
    num_params = ansatz.num_parameters
    x0 = rng.uniform(-np.pi, np.pi, num_params)

    energy_history: list[float] = []

    def objective(params: np.ndarray) -> float:
        """Evaluate ⟨ψ(θ)|H|ψ(θ)⟩."""
        bound_circuit = ansatz.assign_parameters(params)
        sv = Statevector(bound_circuit)
        energy = float(sv.expectation_value(H).real)
        energy_history.append(energy)
        return energy

    # Run optimization
    if optimizer == "L-BFGS-B":
        result = minimize(
            objective, x0, method=optimizer,
            options={'maxiter': maxiter, 'maxfun': maxiter * 3},
        )
    else:
        result = minimize(
            objective, x0, method=optimizer,
            options={'maxiter': maxiter},
        )

    return {
        'energy': float(result.fun),
        'params': result.x,
        'history': energy_history,
        'num_evals': result.nfev,
    }
