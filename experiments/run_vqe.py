"""VQE experiment: convergence curves, error analysis across configurations."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsim.vqe import (
    ising_hamiltonian,
    hardware_efficient_ansatz,
    run_vqe,
    exact_ground_energy,
)


def run_vqe_sweep() -> dict:
    """Run VQE across n, h/J ratios, and layer counts."""
    results = {}

    ns = [2, 3, 4, 5, 6]
    h_ratios = [0.5, 1.0, 1.5]
    layer_counts = [1, 2, 3]

    for n in ns:
        for h_over_J in h_ratios:
            H = ising_hamiltonian(n, J=1.0, h=h_over_J)
            exact_e = exact_ground_energy(H)

            for layers in layer_counts:
                ansatz = hardware_efficient_ansatz(n, layers)

                # Multiple seeds
                best_energy = float('inf')
                mean_energies = []

                for seed in range(5):
                    res = run_vqe(H, ansatz, optimizer="COBYLA",
                                  seed=seed * 17 + 42, maxiter=500)
                    mean_energies.append(res['energy'])
                    if res['energy'] < best_energy:
                        best_energy = res['energy']
                        best_result = res

                abs_err = abs(best_energy - exact_e)
                rel_err = abs_err / abs(exact_e) if abs(exact_e) > 1e-10 else abs_err

                key = f"n={n},h/J={h_over_J},L={layers}"
                results[key] = {
                    'n': n, 'h_over_J': h_over_J, 'layers': layers,
                    'exact_energy': exact_e,
                    'vqe_best_energy': best_energy,
                    'vqe_mean_energy': float(np.mean(mean_energies)),
                    'abs_error': abs_err,
                    'rel_error': rel_err,
                    'history': best_result['history'],
                    'num_evals': best_result['num_evals'],
                }

                status = "✓" if rel_err < 0.01 else "✗"
                print(f"  {status} {key}: VQE={best_energy:.4f}, "
                      f"exact={exact_e:.4f}, rel_err={rel_err:.2e}")

    return results


def plot_convergence_curves(results: dict) -> None:
    """Plot convergence curves for selected configurations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    configs = [
        ('n=2,h/J=1.0,L=2', 'n=2, h/J=1.0, L=2'),
        ('n=3,h/J=1.0,L=2', 'n=3, h/J=1.0, L=2'),
        ('n=4,h/J=1.0,L=2', 'n=4, h/J=1.0, L=2'),
        ('n=4,h/J=0.5,L=3', 'n=4, h/J=0.5, L=3'),
        ('n=5,h/J=1.0,L=3', 'n=5, h/J=1.0, L=3'),
        ('n=6,h/J=1.0,L=3', 'n=6, h/J=1.0, L=3'),
    ]

    for ax, (key, title) in zip(axes.flat, configs):
        if key in results:
            data = results[key]
            history = data['history']
            ax.plot(history, color='#1976d2', alpha=0.7, linewidth=0.8)
            ax.axhline(y=data['exact_energy'], color='red', linestyle='--',
                       label=f"Exact: {data['exact_energy']:.3f}")
            ax.set_xlabel('Function Evaluation')
            ax.set_ylabel('Energy')
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/vqe_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/vqe_convergence.png")


def plot_error_vs_layers(results: dict) -> None:
    """Plot VQE error vs number of layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Error vs layers for different n (h/J=1.0)
    for n in [2, 3, 4, 5, 6]:
        errors = []
        layers_list = [1, 2, 3]
        for L in layers_list:
            key = f"n={n},h/J=1.0,L={L}"
            if key in results:
                errors.append(results[key]['rel_error'])
            else:
                errors.append(None)

        valid = [(l, e) for l, e in zip(layers_list, errors) if e is not None]
        if valid:
            ls, es = zip(*valid)
            ax1.semilogy(ls, es, 'o-', label=f'n={n}', markersize=6)

    ax1.set_xlabel('Number of Layers', fontsize=12)
    ax1.set_ylabel('Relative Error', fontsize=12)
    ax1.set_title('VQE Error vs Ansatz Layers (h/J=1.0)', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1, 2, 3])

    # Error vs n for different layers (h/J=1.0)
    for L in [1, 2, 3]:
        errors = []
        ns = [2, 3, 4, 5, 6]
        for n in ns:
            key = f"n={n},h/J=1.0,L={L}"
            if key in results:
                errors.append(results[key]['rel_error'])
            else:
                errors.append(None)

        valid = [(nn, e) for nn, e in zip(ns, errors) if e is not None]
        if valid:
            nns, es = zip(*valid)
            ax2.semilogy(nns, es, 'o-', label=f'L={L}', markersize=6)

    ax2.set_xlabel('Number of Qubits (n)', fontsize=12)
    ax2.set_ylabel('Relative Error', fontsize=12)
    ax2.set_title('VQE Error vs Qubits (h/J=1.0)', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.01, color='gray', linestyle=':', alpha=0.5, label='1% threshold')

    plt.tight_layout()
    plt.savefig('results/vqe_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/vqe_error_analysis.png")


def main() -> dict:
    """Run all VQE experiments."""
    print("=" * 60)
    print("VQE Experiments")
    print("=" * 60)

    print("\n1. VQE Sweep (n, h/J, layers):")
    results = run_vqe_sweep()

    print("\n2. Generating convergence plots...")
    plot_convergence_curves(results)

    print("\n3. Generating error analysis plots...")
    plot_error_vs_layers(results)

    # Summary table for key configs
    summary = {}
    for key, data in results.items():
        summary[key] = {
            'exact_energy': data['exact_energy'],
            'vqe_best_energy': data['vqe_best_energy'],
            'abs_error': data['abs_error'],
            'rel_error': data['rel_error'],
        }

    return {'vqe_results': summary}


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    results = main()
    print("\nDone.")
