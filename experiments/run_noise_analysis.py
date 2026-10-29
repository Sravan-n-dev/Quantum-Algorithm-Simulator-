"""Noise analysis experiments: depolarizing error, readout error, depth interaction,

QAOA noise crossover, Shor under noise, and calibration-matrix error mitigation.
"""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from qsim.grover import (
    grover_circuit,
    optimal_iterations,
    analytical_success_probability,
)
from qsim.qaoa import (
    qaoa_circuit,
    run_qaoa,
    evaluate_cut,
    brute_force_maxcut,
)
from qsim.shor import (
    order_finding_circuit,
    extract_order,
)
from qsim.noise import (
    depolarizing_noise_model,
    readout_noise_model,
    combined_noise_model,
    run_with_noise,
    success_probability,
    binomial_std_error,
    mitigate_readout_error,
    NOISE_BASIS_GATES,
)


def run_grover_vs_depolarizing() -> dict:
    """Analyze Grover success probability vs two-qubit depolarizing error rate."""
    print("\n--- 1. Grover Success vs 2-Qubit Depolarizing Noise (p2) ---")
    results = {}
    p2_vals = [0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
    shots = 4096

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {3: '#1976d2', 4: '#d32f2f', 5: '#388e3c'}
    markers = {3: 'o', 4: 's', 5: '^'}

    for n in [3, 4, 5]:
        marked = ['0' * (n - 1) + '1']
        k_opt = optimal_iterations(n, 1)
        qc = grover_circuit(n, marked, iterations=k_opt)
        qc.measure_all()

        # Get transpiled CX count for simple analytical decay model: F ≈ (1 - p2)^N_cx
        transpiled = transpile(qc, basis_gates=NOISE_BASIS_GATES, optimization_level=0)
        n_cx = transpiled.count_ops().get('cx', 0)
        p_ideal = analytical_success_probability(n, 1, k_opt)

        probs = []
        errs = []
        for p2 in p2_vals:
            # p1 = p2 / 10 is typical in superconducting transmon hardware
            model = depolarizing_noise_model(p1=p2 / 10, p2=p2)
            counts = run_with_noise(qc, model, shots=shots, seed=42)
            p = success_probability(counts, marked)
            err = binomial_std_error(p, shots)
            probs.append(p)
            errs.append(err)

        results[str(n)] = {
            'p2_values': p2_vals,
            'success_probs': probs,
            'std_errors': errs,
            'n_cx': n_cx,
            'p_ideal': p_ideal,
        }

        # Plot data with error bars
        ax.errorbar(p2_vals, probs, yerr=errs, fmt=markers[n] + '-', color=colors[n],
                    label=f'n={n} (simulated, {n_cx} CX)', capsize=3, markersize=6, alpha=0.9)
        # Analytical simple decay curve: p_ideal * (1 - p2)^N_cx + (1/2^n)(1 - (1-p2)^N_cx)
        decay_curve = [p_ideal * ((1 - p) ** n_cx) + (1.0 / (2 ** n)) * (1.0 - (1 - p) ** n_cx)
                       for p in p2_vals]
        ax.plot(p2_vals, decay_curve, '--', color=colors[n], alpha=0.4,
                label=f'n={n} analytical model: $(1-p_2)^{{{n_cx}}}$')
        # Ideal horizontal line
        ax.axhline(y=p_ideal, color=colors[n], linestyle=':', alpha=0.3)

        print(f"  n={n} (CX={n_cx}): p2=0.0 -> P={probs[0]:.4f}, p2=0.05 -> P={probs[-1]:.4f}")

    ax.set_xlabel('Two-Qubit Depolarizing Error Rate ($p_2$)', fontsize=12)
    ax.set_ylabel('Success Probability', fontsize=12)
    ax.set_title("Grover's Algorithm: Success Probability vs Depolarizing Noise ($p_2$)", fontsize=14)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/noise_grover_vs_p2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/noise_grover_vs_p2.png")

    return results


def run_grover_depth_interaction() -> dict:
    """Analyze how noise interacts with circuit depth across Grover iterations."""
    print("\n--- 2. Grover Success vs Circuit Depth under Noise ---")
    n = 3
    marked = ['001']
    k_opt = optimal_iterations(n, 1)  # k_opt = 2
    iterations_list = list(range(1, k_opt + 4))  # 1 to 5 iterations

    noise_levels = [0.0, 0.002, 0.008, 0.02]
    colors = ['#1976d2', '#388e3c', '#f57c00', '#d32f2f']
    shots = 4096

    results = {}
    depths = []
    cx_counts = []

    # Precompute depths
    for k in iterations_list:
        qc = grover_circuit(n, marked, iterations=k)
        qc.measure_all()
        transpiled = transpile(qc, basis_gates=NOISE_BASIS_GATES, optimization_level=0)
        depths.append(transpiled.depth())
        cx_counts.append(transpiled.count_ops().get('cx', 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, p2 in enumerate(noise_levels):
        probs = []
        for k in iterations_list:
            qc = grover_circuit(n, marked, iterations=k)
            qc.measure_all()
            model = depolarizing_noise_model(p1=p2 / 10, p2=p2)
            counts = run_with_noise(qc, model, shots=shots, seed=42)
            probs.append(success_probability(counts, marked))

        results[f"p2={p2}"] = {
            'iterations': iterations_list,
            'depths': depths,
            'cx_counts': cx_counts,
            'probs': probs,
        }

        label = "Ideal ($p_2=0$)" if p2 == 0.0 else f"$p_2={p2}$"
        ax1.plot(iterations_list, probs, 'o-', color=colors[idx], linewidth=2,
                 markersize=6, label=label)
        ax2.plot(depths, probs, 's-', color=colors[idx], linewidth=2,
                 markersize=6, label=label)

        print(f"  Noise p2={p2}: probs at k=1..{len(iterations_list)}: {[round(x, 4) for x in probs]}")

    ax1.set_xlabel('Grover Iterations (k)', fontsize=12)
    ax1.set_ylabel('Success Probability', fontsize=12)
    ax1.set_title(f'Grover (n={n}): Success vs Iterations Under Noise', fontsize=13)
    ax1.axvline(x=k_opt, color='gray', linestyle=':', label=f'Optimal k={k_opt}')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Transpiled Circuit Depth', fontsize=12)
    ax2.set_ylabel('Success Probability', fontsize=12)
    ax2.set_title(f'Grover (n={n}): Success vs Circuit Depth Under Noise', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/noise_grover_depth_interaction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/noise_grover_depth_interaction.png")

    return results


def run_readout_error_and_mitigation() -> dict:
    """Analyze readout error sweep and test calibration-matrix inversion mitigation."""
    print("\n--- 3. Readout Error Sweep & Error Mitigation ---")
    p_ro_vals = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    shots = 4096
    results = {}

    # Grover test
    n_g = 3
    marked = ['101']
    k_opt = optimal_iterations(n_g, 1)
    qc_g = grover_circuit(n_g, marked, iterations=k_opt)
    qc_g.measure_all()

    grover_raw = []
    grover_mitigated = []

    # QAOA test on 4-cycle
    g_cycle = nx.cycle_graph(4)
    qaoa_res = run_qaoa(g_cycle, p=1, optimizer="COBYLA", seed=42, restarts=2)
    qc_qaoa = qaoa_circuit(g_cycle, qaoa_res['gammas'], qaoa_res['betas'])
    qc_qaoa.measure_all()
    opt_strings = qaoa_res['optimal_strings']

    qaoa_raw = []
    qaoa_mitigated = []

    for p_ro in p_ro_vals:
        model = readout_noise_model(p_ro, p_ro)

        # Grover run
        counts_g = run_with_noise(qc_g, model, shots=shots, seed=42)
        raw_g = success_probability(counts_g, marked)
        mit_dict_g = mitigate_readout_error(counts_g, p_ro, p_ro, num_qubits=n_g)
        mit_g = sum(mit_dict_g.get(bs, 0.0) for bs in marked)
        grover_raw.append(raw_g)
        grover_mitigated.append(mit_g)

        # QAOA run
        counts_q = run_with_noise(qc_qaoa, model, shots=shots, seed=42)
        raw_q = success_probability(counts_q, opt_strings)
        mit_dict_q = mitigate_readout_error(counts_q, p_ro, p_ro, num_qubits=4)
        mit_q = sum(mit_dict_q.get(bs, 0.0) for bs in opt_strings)
        qaoa_raw.append(raw_q)
        qaoa_mitigated.append(mit_q)

        print(f"  p_ro={p_ro:.2f}: Grover raw={raw_g:.4f} -> mit={mit_g:.4f} | "
              f"QAOA raw={raw_q:.4f} -> mit={mit_q:.4f}")

    results['p_ro_values'] = p_ro_vals
    results['grover_raw'] = grover_raw
    results['grover_mitigated'] = grover_mitigated
    results['qaoa_raw'] = qaoa_raw
    results['qaoa_mitigated'] = qaoa_mitigated

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(p_ro_vals, grover_raw, 'o-', color='#d32f2f', linewidth=2, markersize=6,
             label='Raw Measured (Noisy)')
    ax1.plot(p_ro_vals, grover_mitigated, 's--', color='#1976d2', linewidth=2, markersize=6,
             label='Matrix-Inverted Mitigated')
    ax1.axhline(y=grover_raw[0], color='green', linestyle=':', label='Ideal (Zero Noise)')
    ax1.set_xlabel('Readout Error Rate ($p_{01} = p_{10}$)', fontsize=12)
    ax1.set_ylabel('Success Probability', fontsize=12)
    ax1.set_title("Grover (n=3): Readout Error & Mitigation", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(p_ro_vals, qaoa_raw, 'o-', color='#d32f2f', linewidth=2, markersize=6,
             label='Raw Measured (Noisy)')
    ax2.plot(p_ro_vals, qaoa_mitigated, 's--', color='#1976d2', linewidth=2, markersize=6,
             label='Matrix-Inverted Mitigated')
    ax2.axhline(y=qaoa_raw[0], color='green', linestyle=':', label='Ideal (Zero Noise)')
    ax2.set_xlabel('Readout Error Rate ($p_{01} = p_{10}$)', fontsize=12)
    ax2.set_ylabel('P(Optimal Cut Bitstring)', fontsize=12)
    ax2.set_title("QAOA 4-Cycle (p=1): Readout Error & Mitigation", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/noise_readout_and_mitigation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/noise_readout_and_mitigation.png")

    return results


def run_qaoa_noise_crossover() -> dict:
    """Analyze QAOA approximation ratio under noise for p=1,2,3 to observe noise crossover."""
    print("\n--- 4. QAOA Noise Crossover: p=1, 2, 3 under Depolarizing Noise ---")
    g = nx.cycle_graph(4)  # 4-cycle Max-Cut (optimal cut = 4)
    opt_cut, _ = brute_force_maxcut(g)

    # First optimize parameters ideally for p=1, 2, 3
    params_per_p = {}
    for p in [1, 2, 3]:
        res = run_qaoa(g, p=p, optimizer="COBYLA", seed=42, restarts=4)
        params_per_p[p] = (res['gammas'], res['betas'])

    p2_vals = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
    shots = 4096
    results = {}

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {1: '#1976d2', 2: '#f57c00', 3: '#d32f2f'}
    markers = {1: 'o', 2: 's', 3: '^'}

    sim_ideal = AerSimulator()
    depth_info = {}
    for p in [1, 2, 3]:
        gammas, betas = params_per_p[p]
        qc = qaoa_circuit(g, gammas, betas)
        qc.measure_all()
        transpiled = transpile(qc, basis_gates=NOISE_BASIS_GATES, optimization_level=0)
        n_cx = transpiled.count_ops().get('cx', 0)
        depth_info[p] = {'depth': transpiled.depth(), 'cx': n_cx}

    for p in [1, 2, 3]:
        gammas, betas = params_per_p[p]
        qc = qaoa_circuit(g, gammas, betas)
        qc.measure_all()

        ratios = []
        for p2 in p2_vals:
            model = depolarizing_noise_model(p1=p2 / 10, p2=p2)
            counts = run_with_noise(qc, model, shots=shots, seed=42)
            # Evaluate expected cut from noisy measurement samples
            exp_cut = sum(counts.get(bs, 0) * evaluate_cut(bs, g) for bs in counts) / shots
            ratio = exp_cut / opt_cut
            ratios.append(ratio)

        results[f"p={p}"] = {
            'ratios': ratios,
            'depth': depth_info[p]['depth'],
            'n_cx': depth_info[p]['cx'],
        }

        label = f"QAOA p={p} (depth={depth_info[p]['depth']}, {depth_info[p]['cx']} CX)"
        ax.plot(p2_vals, ratios, marker=markers[p], linestyle='-', color=colors[p],
                linewidth=2, markersize=6, label=label)
        print(f"  p={p} (CX={depth_info[p]['cx']}): p2=0.0 -> ratio={ratios[0]:.4f}, "
              f"p2=0.10 -> ratio={ratios[-1]:.4f}")

    # Theoretical random guess cut for 4-cycle: expected cut is 2 (ratio = 0.5)
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Random Guess (Ratio = 0.50)')

    ax.set_xlabel('Two-Qubit Depolarizing Error Rate ($p_2$)', fontsize=12)
    ax.set_ylabel('Approximation Ratio', fontsize=12)
    ax.set_title('QAOA Max-Cut under Depolarizing Noise: Depth vs Noise Crossover', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/noise_qaoa_crossover.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/noise_qaoa_crossover.png")

    results['p2_values'] = p2_vals
    return results


def run_shor_under_noise() -> dict:
    """Analyze small Shor order finding (N=15, a=7) under depolarizing noise."""
    print("\n--- 5. Shor Order Finding (N=15, a=7) under Depolarizing Noise ---")
    N, a = 15, 7
    r_expected = 4
    t = 8  # 8 counting qubits, 4 target qubits = 12 total qubits
    expected_peaks = [0, 64, 128, 192]

    qc = order_finding_circuit(a, N)
    p2_vals = [0.0, 0.001, 0.003, 0.007, 0.015]
    shots = 2048

    results = {}
    peak_probs = []
    order_success_rates = []

    for p2 in p2_vals:
        model = depolarizing_noise_model(p1=p2 / 10, p2=p2)
        counts = run_with_noise(qc, model, shots=shots, seed=42)

        # Calculate peak probability
        peak_shots = sum(
            counts.get(f"{p:0{t}b}", 0) for p in expected_peaks
        )
        p_peak = peak_shots / shots
        peak_probs.append(p_peak)

        # Check order extraction rate from sample outcomes
        valid_orders = 0
        for bs, c in counts.items():
            measured_int = int(bs, 2)
            r_cand = extract_order(measured_int, t, N, a)
            if r_cand == r_expected:
                valid_orders += c
        order_rate = valid_orders / shots
        order_success_rates.append(order_rate)

        print(f"  p2={p2:.3f}: peak_prob={p_peak:.4f}, correct_order_rate={order_rate:.4f}")

    results['p2_values'] = p2_vals
    results['peak_probs'] = peak_probs
    results['order_success_rates'] = order_success_rates

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(p2_vals, peak_probs, 'o-', color='#1976d2', linewidth=2, markersize=6,
            label='P(Measurement in Peak States {0,64,128,192})')
    ax.plot(p2_vals, order_success_rates, 's--', color='#d32f2f', linewidth=2, markersize=6,
            label='Correct Order Extraction Rate (r=4)')

    ax.set_xlabel('Two-Qubit Depolarizing Error Rate ($p_2$)', fontsize=12)
    ax.set_ylabel('Probability / Success Rate', fontsize=12)
    ax.set_title('Shor Order Finding ($N=15, a=7$, 12 Qubits) under Noise', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/noise_shor_degradation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/noise_shor_degradation.png")

    return results


def main() -> dict:
    """Run all noise experiments."""
    print("=" * 60)
    print("Quantum Hardware Noise Simulation & Analysis")
    print("=" * 60)

    grover_res = run_grover_vs_depolarizing()
    depth_res = run_grover_depth_interaction()
    readout_res = run_readout_error_and_mitigation()
    qaoa_res = run_qaoa_noise_crossover()
    shor_res = run_shor_under_noise()

    return {
        'grover_noise': grover_res,
        'grover_depth': depth_res,
        'readout_mitigation': readout_res,
        'qaoa_noise': qaoa_res,
        'shor_noise': shor_res,
    }


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    res = main()
    print("\nAll noise experiments complete.")
