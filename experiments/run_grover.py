"""Grover experiment: success probability vs iterations, circuit complexity."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsim.grover import (
    grover_circuit,
    analytical_success_probability,
    optimal_iterations,
)


def run_success_vs_iterations() -> dict:
    """Measure success probability vs iterations for various n."""
    results = {}

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1976d2', '#d32f2f', '#388e3c', '#f57c00']

    for idx, n in enumerate([3, 4, 5, 6]):
        M = 1
        marked = ['0' * (n - 1) + '1']
        k_opt = optimal_iterations(n, M)
        k_max = k_opt + 3

        # Simulated points
        sim_probs = []
        analytical_probs = []
        ks = list(range(k_max + 1))

        for k in ks:
            qc = grover_circuit(n, marked, iterations=k)
            sv = Statevector(qc)
            prob = sum(sv.probabilities_dict().get(bs, 0.0) for bs in marked)
            sim_probs.append(prob)
            analytical_probs.append(analytical_success_probability(n, M, k))

        results[n] = {
            'optimal_k': k_opt,
            'success_at_optimal': sim_probs[k_opt],
            'analytical_at_optimal': analytical_probs[k_opt],
        }

        # Plot
        k_fine = np.linspace(0, k_max, 200)
        analytical_fine = [analytical_success_probability(n, M, kk) for kk in k_fine]

        ax.plot(k_fine, analytical_fine, '-', color=colors[idx], alpha=0.5,
                label=f'n={n} analytical')
        ax.plot(ks, sim_probs, 'o', color=colors[idx], markersize=6,
                label=f'n={n} simulated')
        ax.axvline(x=k_opt, color=colors[idx], linestyle=':', alpha=0.3)

        print(f"  n={n}: optimal k={k_opt}, P(success)={sim_probs[k_opt]:.6f}")

    ax.set_xlabel('Number of Iterations (k)', fontsize=12)
    ax.set_ylabel('Success Probability', fontsize=12)
    ax.set_title("Grover's Algorithm: Success Probability vs Iterations", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/grover_success_vs_iterations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/grover_success_vs_iterations.png")

    return results


def run_circuit_complexity() -> dict:
    """Measure circuit depth and gate count vs n."""
    from qiskit_aer import AerSimulator

    results = {}
    ns = list(range(2, 9))
    depths = []
    cx_counts = []
    total_gates = []

    for n in ns:
        marked = ['0' * (n - 1) + '1']
        k_opt = optimal_iterations(n, 1)
        qc = grover_circuit(n, marked, iterations=k_opt)

        sim = AerSimulator()
        transpiled = transpile(qc, sim, basis_gates=['u', 'cx'],
                               optimization_level=0)
        d = transpiled.depth()
        ops = transpiled.count_ops()
        cx = ops.get('cx', 0)
        total = sum(ops.values())

        depths.append(d)
        cx_counts.append(cx)
        total_gates.append(total)

        results[n] = {
            'depth': d, 'cx_gates': cx, 'total_gates': total,
            'optimal_k': k_opt,
        }
        print(f"  n={n}: depth={d}, CX={cx}, total={total}, k_opt={k_opt}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ns, depths, 'o-', color='#1976d2', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Qubits (n)', fontsize=12)
    ax1.set_ylabel('Circuit Depth', fontsize=12)
    ax1.set_title('Grover Circuit Depth vs Qubits', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ns, cx_counts, 's-', color='#d32f2f', linewidth=2, markersize=6,
             label='CX gates')
    ax2.plot(ns, total_gates, 'o-', color='#1976d2', linewidth=2, markersize=6,
             label='Total gates')
    ax2.set_xlabel('Number of Qubits (n)', fontsize=12)
    ax2.set_ylabel('Gate Count', fontsize=12)
    ax2.set_title('Grover Gate Count vs Qubits', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/grover_circuit_complexity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/grover_circuit_complexity.png")

    return results


def main() -> dict:
    """Run all Grover experiments."""
    print("=" * 60)
    print("Grover Experiments")
    print("=" * 60)

    print("\n1. Success Probability vs Iterations:")
    success_results = run_success_vs_iterations()

    print("\n2. Circuit Complexity:")
    complexity_results = run_circuit_complexity()

    return {
        'grover_success': {str(k): v for k, v in success_results.items()},
        'grover_complexity': {str(k): v for k, v in complexity_results.items()},
    }


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    results = main()
    print("\nDone.")
