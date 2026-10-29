"""Master benchmark runner: executes all experiments, saves plots, and dumps results.json."""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import experiments.run_grover as exp_grover
import experiments.run_qft as exp_qft
import experiments.run_shor as exp_shor
import experiments.run_vqe as exp_vqe
import experiments.run_qaoa as exp_qaoa
import experiments.run_noise_analysis as exp_noise


def main() -> None:
    """Run all experiments and record consolidated results."""
    os.makedirs('results', exist_ok=True)
    start_total = time.time()

    print("=" * 70)
    print("QUANTUM ALGORITHM SIMULATOR & BENCHMARK SUITE — MASTER BENCHMARK RUN")
    print("=" * 70)

    all_results: dict[str, dict] = {}

    # 1. Grover Experiments
    print("\n" + "#" * 50)
    print("# 1. Running Grover Search Experiments")
    print("#" * 50)
    t0 = time.time()
    all_results.update(exp_grover.main())
    print(f"  [Grover completed in {time.time() - t0:.2f}s]")

    # 2. QFT Experiments
    print("\n" + "#" * 50)
    print("# 2. Running Quantum Fourier Transform Experiments")
    print("#" * 50)
    t0 = time.time()
    all_results.update(exp_qft.main())
    print(f"  [QFT completed in {time.time() - t0:.2f}s]")

    # 3. Shor Experiments
    print("\n" + "#" * 50)
    print("# 3. Running Shor Factoring Experiments")
    print("#" * 50)
    t0 = time.time()
    all_results.update(exp_shor.main())
    print(f"  [Shor completed in {time.time() - t0:.2f}s]")

    # 4. VQE Experiments
    print("\n" + "#" * 50)
    print("# 4. Running VQE Experiments")
    print("#" * 50)
    t0 = time.time()
    all_results.update(exp_vqe.main())
    print(f"  [VQE completed in {time.time() - t0:.2f}s]")

    # 5. QAOA Experiments
    print("\n" + "#" * 50)
    print("# 5. Running QAOA Max-Cut Experiments")
    print("#" * 50)
    t0 = time.time()
    all_results.update(exp_qaoa.main())
    print(f"  [QAOA completed in {time.time() - t0:.2f}s]")

    # 6. Noise Analysis Experiments
    print("\n" + "#" * 50)
    print("# 6. Running Hardware Noise Simulation & Analysis")
    print("#" * 50)
    t0 = time.time()
    all_results.update(exp_noise.main())
    print(f"  [Noise Analysis completed in {time.time() - t0:.2f}s]")

    total_time = time.time() - start_total
    all_results['metadata'] = {
        'total_runtime_seconds': total_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Save results.json
    results_path = os.path.join('results', 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved consolidated benchmarks to {results_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("KEY MEASURED BENCHMARK SUMMARY (TRUTHFUL RECORD)")
    print("=" * 70)

    print("\n--- GROVER SUCCESS AT OPTIMAL ITERATIONS (M=1) ---")
    for n_str, data in all_results.get('grover_success', {}).items():
        print(f"  n={n_str}: optimal_k={data['optimal_k']}, "
              f"P(sim)={data['success_at_optimal']:.6f}, "
              f"P(theory)={data['analytical_at_optimal']:.6f}")

    print("\n--- QFT VS DFT OPERATOR EQUIVALENCE ---")
    for n_str, data in all_results.get('qft_verification', {}).items():
        print(f"  n={n_str}: Operator.equiv={data['equivalent']}, "
              f"Norm Error={data['norm_error']:.2e}")

    print("\n--- SHOR FACTORING SUCCESS (50 SEEDS) ---")
    for n_str, data in all_results.get('shor_factoring', {}).items():
        print(f"  N={n_str}: Success Rate={data['success_rate']:.2%}, "
              f"Avg Attempts={data['avg_attempts']:.2f}, Factors={data['factors_found']}")

    print("\n--- VQE GROUND STATE RELATIVE ERRORS (Selected) ---")
    for k, v in all_results.get('vqe_results', {}).items():
        if 'h/J=1.0' in k:
            print(f"  {k:22s}: Exact={v['exact_energy']:.4f}, "
                  f"VQE={v['vqe_best_energy']:.4f}, RelError={v['rel_error']:.2e}")

    print("\n--- QAOA MAX-CUT APPROXIMATION RATIOS ---")
    for g_name, g_data in all_results.get('qaoa_results', {}).items():
        p_str = ", ".join(f"p={p}: {res['approx_ratio']:.4f}"
                          for p, res in g_data.get('p_results', {}).items())
        print(f"  {g_name:25s} (OptCut={g_data['optimal_cut']}): {p_str}")

    print(f"\nAll experiments successfully completed in {total_time:.2f}s ({total_time / 60:.1f} min)")
    print("=" * 70)


if __name__ == '__main__':
    main()
