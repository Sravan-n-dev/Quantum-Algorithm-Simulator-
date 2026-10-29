"""Shor experiment: order finding histograms and factoring success rates."""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qsim.shor import (
    order_finding_circuit,
    classical_order,
    extract_order,
    shor_factor,
)


def run_order_finding_histogram() -> dict:
    """Plot the counting register histogram for N=15, a=7."""
    N, a = 15, 7
    r_expected = 4
    t = 2 * math.ceil(math.log2(N))  # t = 8

    qc = order_finding_circuit(a, N)
    sim = AerSimulator(method='statevector')
    transpiled = transpile(qc, sim)
    result = sim.run(transpiled, shots=4096, seed_simulator=42).result()
    counts = result.get_counts()

    # Expected peaks at s * 2^t / r = s * 64 for s = 0,1,2,3
    expected_peaks = [s * (2 ** t) // r_expected for s in range(r_expected)]

    print(f"  N={N}, a={a}, r={r_expected}, t={t}")
    print(f"  Expected peaks at: {expected_peaks}")

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    x_vals = sorted(counts.keys(), key=lambda bs: int(bs, 2))
    x_ints = [int(bs, 2) for bs in x_vals]
    y_vals = [counts[bs] for bs in x_vals]

    colors = ['#d32f2f' if x in expected_peaks else '#1976d2' for x in x_ints]
    ax.bar(x_ints, y_vals, color=colors, alpha=0.8, width=2.0)

    for p in expected_peaks:
        ax.axvline(x=p, color='red', linestyle='--', alpha=0.5)
        ax.annotate(f'{p}\n(s/r={p * r_expected // (2**t)}/{r_expected})',
                    xy=(p, max(y_vals) * 0.9), fontsize=8, ha='center',
                    color='red')

    ax.set_xlabel('Measured Value', fontsize=12)
    ax.set_ylabel('Counts (out of 4096)', fontsize=12)
    ax.set_title(f"Order Finding: N={N}, a={a} (r={r_expected}, t={t})", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/shor_order_finding_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/shor_order_finding_histogram.png")

    # Top counts
    top = sorted(counts.items(), key=lambda x: -x[1])[:10]
    result_data = {
        'N': N, 'a': a, 'r_expected': r_expected, 't': t,
        'expected_peaks': expected_peaks,
        'top_counts': {bs: c for bs, c in top},
        'total_shots': 4096,
    }
    for bs, c in top:
        val = int(bs, 2)
        print(f"    {bs} (={val:3d}): {c:4d} shots  {'<-- peak' if val in expected_peaks else ''}")

    return result_data


def run_factoring_success_rate() -> dict:
    """Run shor_factor across 50 seeds and report success rates."""
    results = {}

    for N in [15, 21]:
        successes = 0
        attempts_list = []
        factors_found = {}

        for seed in range(50):
            try:
                p, q, attempts = shor_factor(N, seed=seed, max_attempts=20)
                if p * q == N and p > 1 and q > 1:
                    successes += 1
                    attempts_list.append(attempts)
                    key = tuple(sorted([p, q]))
                    factors_found[str(key)] = factors_found.get(str(key), 0) + 1
            except ValueError:
                pass

        success_rate = successes / 50
        avg_attempts = np.mean(attempts_list) if attempts_list else float('nan')

        results[N] = {
            'success_rate': success_rate,
            'successes_out_of_50': successes,
            'avg_attempts': float(avg_attempts),
            'factors_found': factors_found,
        }
        print(f"  N={N}: {successes}/50 seeds succeeded "
              f"(rate={success_rate:.2%}), avg attempts={avg_attempts:.1f}")
        print(f"    Factors found: {factors_found}")

    return results


def main() -> dict:
    """Run all Shor experiments."""
    print("=" * 60)
    print("Shor Experiments")
    print("=" * 60)

    print("\n1. Order Finding Histogram (N=15, a=7):")
    hist_results = run_order_finding_histogram()

    print("\n2. Factoring Success Rate (50 seeds):")
    factor_results = run_factoring_success_rate()

    return {
        'shor_histogram': hist_results,
        'shor_factoring': {str(k): v for k, v in factor_results.items()},
    }


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    results = main()
    print("\nDone.")
