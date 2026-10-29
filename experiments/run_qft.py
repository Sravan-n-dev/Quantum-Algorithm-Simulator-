"""QFT experiment: verify against DFT matrix and demonstrate period detection."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsim.qft import qft_circuit, dft_matrix


def verify_qft_vs_dft() -> dict:
    """Verify QFT matches DFT matrix for n=1..8."""
    results = {}
    for n in range(1, 9):
        qc = qft_circuit(n)
        qft_op = Operator(qc)
        dft = dft_matrix(n)

        # Check equivalence up to global phase
        equiv = qft_op.equiv(Operator(dft), atol=1e-8)

        # Compute norm of difference (for reporting)
        diff = qft_op.data - dft
        # Factor out global phase
        if np.abs(qft_op.data[0, 0]) > 1e-10:
            phase = dft[0, 0] / qft_op.data[0, 0]
            diff = qft_op.data * phase - dft
        norm_diff = float(np.linalg.norm(diff))

        results[n] = {'equivalent': equiv, 'norm_error': norm_diff}
        print(f"  n={n}: equiv={equiv}, ||error||={norm_diff:.2e}")

    return results


def period_detection_demo() -> dict:
    """Demonstrate QFT period detection on periodic states."""
    results = {}

    for n, r in [(6, 4), (7, 4), (8, 8)]:
        N = 2 ** n
        # Create periodic state: |0⟩ + |r⟩ + |2r⟩ + ...
        qc = QuantumCircuit(n)
        # Prepare periodic state via superposition of states at multiples of r
        num_periods = N // r
        state = np.zeros(N, dtype=complex)
        for k in range(num_periods):
            state[k * r] = 1.0
        state /= np.linalg.norm(state)

        qc.initialize(state, range(n))
        qc.compose(qft_circuit(n), inplace=True)

        sv = Statevector(qc)
        probs = sv.probabilities()

        # Find peaks (should be at multiples of N/r)
        expected_peak_spacing = N // r
        peak_indices = np.where(probs > 0.01)[0]

        results[f"n={n},r={r}"] = {
            'peak_indices': peak_indices.tolist(),
            'expected_spacing': expected_peak_spacing,
            'peak_probs': {str(i): float(probs[i]) for i in peak_indices},
        }

        print(f"  n={n}, r={r}: peaks at {peak_indices.tolist()}, "
              f"expected spacing N/r={expected_peak_spacing}")

    return results


def plot_period_detection() -> None:
    """Plot period detection probability distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    configs = [(6, 4), (7, 4), (8, 8)]
    for ax, (n, r) in zip(axes, configs):
        N = 2 ** n
        qc = QuantumCircuit(n)
        num_periods = N // r
        state = np.zeros(N, dtype=complex)
        for k in range(num_periods):
            state[k * r] = 1.0
        state /= np.linalg.norm(state)

        qc.initialize(state, range(n))
        qc.compose(qft_circuit(n), inplace=True)

        sv = Statevector(qc)
        probs = sv.probabilities()

        # Highlight expected peaks
        expected_peaks = [k * (N // r) for k in range(r) if k * (N // r) < N]

        colors = ['#d32f2f' if i in expected_peaks else '#1976d2'
                   for i in range(N)]

        ax.bar(range(N), probs, color=colors, alpha=0.8, width=1.0)
        ax.set_xlabel('Measurement Outcome')
        ax.set_ylabel('Probability')
        ax.set_title(f'QFT Period Detection\nn={n}, r={r}, peaks at multiples of {N//r}')

        # Mark expected peaks
        for p in expected_peaks:
            ax.axvline(x=p, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/qft_period_detection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/qft_period_detection.png")


def main() -> dict:
    """Run all QFT experiments."""
    print("=" * 60)
    print("QFT Experiments")
    print("=" * 60)

    print("\n1. QFT vs DFT Matrix Verification:")
    verify_results = verify_qft_vs_dft()

    print("\n2. Period Detection Demo:")
    period_results = period_detection_demo()

    print("\n3. Generating plots...")
    plot_period_detection()

    return {
        'qft_verification': {
            str(k): v for k, v in verify_results.items()
        },
        'period_detection': period_results,
    }


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    results = main()
    print("\nDone.")
