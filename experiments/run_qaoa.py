"""QAOA experiment: approximation ratios across graphs, cut visualization, distributions."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from qsim.qaoa import (
    maxcut_hamiltonian,
    qaoa_circuit,
    run_qaoa,
    brute_force_maxcut,
    evaluate_cut,
)


def get_benchmark_graphs() -> dict[str, nx.Graph]:
    """Create benchmark graphs for QAOA testing."""
    graphs = {}

    # 1. Triangle (3 nodes)
    graphs['Triangle'] = nx.cycle_graph(3)

    # 2. 4-Cycle (4 nodes)
    graphs['4-Cycle'] = nx.cycle_graph(4)

    # 3. 6-node 3-regular graph
    graphs['3-Regular (n=6)'] = nx.random_regular_graph(3, 6, seed=42)

    # 4. 8-node Erdos-Renyi graph (p=0.5)
    graphs['Erdos-Renyi (n=8, p=0.5)'] = nx.erdos_renyi_graph(8, 0.5, seed=42)

    # 5. Small weighted 5-node graph (5-cycle with weights)
    wg = nx.cycle_graph(5)
    weights = [1.0, 2.5, 1.5, 3.0, 2.0]
    for idx, (u, v) in enumerate(wg.edges()):
        wg[u][v]['weight'] = weights[idx]
    graphs['Weighted 5-Cycle'] = wg

    return graphs


def run_qaoa_sweep() -> dict:
    """Run QAOA for p=1,2,3 across all benchmark graphs."""
    graphs = get_benchmark_graphs()
    results = {}

    for name, g in graphs.items():
        n = g.number_of_nodes()
        opt_cut, opt_strings = brute_force_maxcut(g)
        print(f"\n--- Graph: {name} (nodes={n}, edges={g.number_of_edges()}, opt_cut={opt_cut}) ---")

        graph_results = {
            'nodes': n,
            'edges': g.number_of_edges(),
            'optimal_cut': opt_cut,
            'optimal_strings': opt_strings,
            'p_results': {},
        }

        for p in [1, 2, 3]:
            # Run QAOA with 5 restarts
            res = run_qaoa(g, p=p, optimizer="COBYLA", seed=42, restarts=5, maxiter=300)
            graph_results['p_results'][p] = {
                'expected_cut': res['expected_cut'],
                'approx_ratio': res['approx_ratio'],
                'optimal_prob': res['optimal_prob'],
                'best_bitstring': res['best_bitstring'],
                'gammas': res['gammas'],
                'betas': res['betas'],
                'probabilities': res['probabilities'],
            }
            print(f"  p={p}: expected_cut={res['expected_cut']:.4f}, "
                  f"ratio={res['approx_ratio']:.4f}, "
                  f"P(opt)={res['optimal_prob']:.4f}")

        results[name] = graph_results

    return results


def plot_approx_ratios(results: dict) -> None:
    """Plot approximation ratio vs p for all graphs."""
    fig, ax = plt.subplots(figsize=(9, 6))

    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1976d2', '#d32f2f', '#388e3c', '#f57c00', '#7b1fa2']

    ps = [1, 2, 3]

    for idx, (name, g_data) in enumerate(results.items()):
        ratios = [g_data['p_results'][p]['approx_ratio'] for p in ps]
        ax.plot(ps, ratios, marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)], linewidth=2, markersize=8,
                label=f"{name}")

    # Theoretical bound line for 3-regular graph at p=1 (0.6924)
    ax.axhline(y=0.6924, color='gray', linestyle='--', alpha=0.7,
               label='3-Regular p=1 theoretical bound (≈0.6924)')

    ax.set_xlabel('QAOA Depth (p)', fontsize=12)
    ax.set_ylabel('Approximation Ratio', fontsize=12)
    ax.set_title('QAOA Max-Cut Approximation Ratio vs Depth (p)', fontsize=14)
    ax.set_xticks(ps)
    ax.set_ylim(0.65, 1.02)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/qaoa_approx_ratio_vs_p.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/qaoa_approx_ratio_vs_p.png")


def plot_graph_cuts(graphs: dict[str, nx.Graph], results: dict) -> None:
    """Draw benchmark graphs with nodes colored by the best cut found."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, (name, g) in zip(axes, graphs.items()):
        n = g.number_of_nodes()
        best_bs = results[name]['p_results'][3]['best_bitstring']
        # In Qiskit little-endian: qubit i corresponds to bitstring[n - 1 - i]
        node_colors = ['#42a5f5' if best_bs[n - 1 - i] == '0' else '#ff7043'
                       for i in range(n)]

        # Determine cut edges vs uncut edges
        cut_edges = []
        uncut_edges = []
        for u, v in g.edges():
            if best_bs[n - 1 - u] != best_bs[n - 1 - v]:
                cut_edges.append((u, v))
            else:
                uncut_edges.append((u, v))

        pos = nx.spring_layout(g, seed=42)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=500, edgecolors='black')
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=uncut_edges, edge_color='gray', width=1.5, style='solid')
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=cut_edges, edge_color='#d32f2f', width=2.5, style='dashed')

        cut_val = evaluate_cut(best_bs, g)
        opt_cut = results[name]['optimal_cut']
        ax.set_title(f"{name}\nCut={cut_val:.1f} / Opt={opt_cut}", fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('results/qaoa_graph_cut.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/qaoa_graph_cut.png")


def plot_bitstring_distributions(results: dict) -> None:
    """Plot probability distribution for 3-Regular graph under p=1,2,3."""
    target_graph = '3-Regular (n=6)'
    if target_graph not in results:
        return

    g_data = results[target_graph]
    opt_strings = set(g_data['optimal_strings'])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for idx, p in enumerate([1, 2, 3]):
        ax = axes[idx]
        probs = g_data['p_results'][p]['probabilities']
        sorted_bs = sorted(probs.keys())
        prob_vals = [probs[bs] for bs in sorted_bs]

        colors = ['#2e7d32' if bs in opt_strings else '#1976d2' for bs in sorted_bs]

        ax.bar(range(len(sorted_bs)), prob_vals, color=colors, alpha=0.85, width=0.8)
        p_opt = g_data['p_results'][p]['optimal_prob']
        ratio = g_data['p_results'][p]['approx_ratio']
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_title(f"QAOA p={p} on {target_graph} — Approx Ratio: {ratio:.4f}, Total P(optimal): {p_opt:.4f}",
                     fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    axes[-1].set_xticks(range(len(sorted_bs)))
    axes[-1].set_xticklabels(sorted_bs, rotation=90, fontsize=7)
    axes[-1].set_xlabel('Computational Basis Bitstring (Green = Optimal Max-Cut)', fontsize=11)

    plt.tight_layout()
    plt.savefig('results/qaoa_bitstring_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved results/qaoa_bitstring_distribution.png")


def main() -> dict:
    """Run all QAOA experiments."""
    print("=" * 60)
    print("QAOA Experiments")
    print("=" * 60)

    graphs = get_benchmark_graphs()
    results = run_qaoa_sweep()

    print("\nGenerating plots...")
    plot_approx_ratios(results)
    plot_graph_cuts(graphs, results)
    plot_bitstring_distributions(results)

    # Prepare serialized summary for results.json
    summary = {}
    for name, g_data in results.items():
        summary[name] = {
            'nodes': g_data['nodes'],
            'edges': g_data['edges'],
            'optimal_cut': g_data['optimal_cut'],
            'p_results': {
                str(p): {
                    'expected_cut': p_res['expected_cut'],
                    'approx_ratio': p_res['approx_ratio'],
                    'optimal_prob': p_res['optimal_prob'],
                    'best_bitstring': p_res['best_bitstring'],
                    'gammas': p_res['gammas'],
                    'betas': p_res['betas'],
                }
                for p, p_res in g_data['p_results'].items()
            }
        }

    return {'qaoa_results': summary}


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    res = main()
    print("\nDone.")
