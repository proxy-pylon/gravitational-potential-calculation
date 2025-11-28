#!/usr/bin/env python3
"""
Plot generation script for Assignment 7: Gravitational Potential Calculation
Generates all required plots from experiment results CSV file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Set up matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.grid'] = True

# Analytical value for validation
U_ANALYTICAL = -0.294524


def load_data(csv_path):
    """Load experiment results from CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiment results from {csv_path}")
    print(f"\nMethods: {df['method'].unique()}")
    print(f"Implementations: {df['implementation'].unique()}")
    print(f"N values: {sorted(df['n_bodies'].unique())}")
    return df


def plot_runtime_vs_n(df, output_dir):
    """
    Plot 1: Runtime vs. N for each method and implementation
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Filter for standard softening value (0.01) and standard sample count (100000 for MC)
    df_plot = df.copy()

    # For Monte Carlo, use only K=100000 to avoid duplicates
    mc_mask = (df_plot['method'] == 'monte_carlo') & (df_plot['n_samples'] == 100000)
    ds_mask = df_plot['method'] == 'direct_summation'
    dl_mask = df_plot['method'] == 'deep_learning'

    df_plot = df_plot[mc_mask | ds_mask | dl_mask]

    # Filter for softening = 0.01
    df_plot = df_plot[df_plot['softening'] == 0.01]

    # Group by method and implementation
    methods = df_plot.groupby(['method', 'implementation'])

    colors = {
        ('direct_summation', 'cuda'): '#1f77b4',
        ('direct_summation', 'cupy'): '#ff7f0e',
        ('monte_carlo', 'cuda'): '#2ca02c',
        ('monte_carlo', 'cupy'): '#d62728',
        ('deep_learning', 'pytorch'): '#9467bd',
        ('deep_learning', 'cupy'): '#8c564b',
    }

    markers = {
        'cuda': 'o',
        'cupy': 's',
        'pytorch': '^',
    }

    for (method, impl), group in methods:
        group = group.sort_values('n_bodies')
        label = f"{method.replace('_', ' ').title()} ({impl.upper()})"
        color = colors.get((method, impl), None)
        marker = markers.get(impl, 'o')

        ax.errorbar(
            group['n_bodies'],
            group['mean_time_s'],
            yerr=group['std_time_s'],
            label=label,
            marker=marker,
            markersize=8,
            capsize=5,
            capthick=2,
            color=color,
            linestyle='-',
            alpha=0.8
        )

    ax.set_xlabel('Number of Particles (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Runtime vs. N for Different Methods and Implementations\n(ε=0.01, K=100,000 for MC)',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'runtime_vs_n.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_speedup_factors(df, output_dir):
    """
    Plot 2: Speedup factors relative to Direct Summation CuPy baseline
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Filter for standard parameters
    df_plot = df.copy()
    mc_mask = (df_plot['method'] == 'monte_carlo') & (df_plot['n_samples'] == 100000)
    ds_mask = df_plot['method'] == 'direct_summation'
    df_plot = df_plot[mc_mask | ds_mask]
    df_plot = df_plot[df_plot['softening'] == 0.01]

    # Get baseline (Direct Summation CuPy)
    baseline = df_plot[(df_plot['method'] == 'direct_summation') &
                       (df_plot['implementation'] == 'cupy')].copy()

    if len(baseline) == 0:
        print("Warning: No baseline data found for Direct Summation CuPy")
        return

    baseline = baseline.set_index('n_bodies')['mean_time_s']

    # Calculate speedup for each method/implementation
    methods = df_plot.groupby(['method', 'implementation'])

    colors = {
        ('direct_summation', 'cuda'): '#1f77b4',
        ('monte_carlo', 'cuda'): '#2ca02c',
        ('monte_carlo', 'cupy'): '#d62728',
    }

    markers = {
        'cuda': 'o',
        'cupy': 's',
    }

    for (method, impl), group in methods:
        # Skip baseline itself
        if method == 'direct_summation' and impl == 'cupy':
            continue

        group = group.sort_values('n_bodies')
        speedups = []
        n_values = []

        for _, row in group.iterrows():
            n = row['n_bodies']
            if n in baseline.index:
                speedup = baseline[n] / row['mean_time_s']
                speedups.append(speedup)
                n_values.append(n)

        if len(speedups) > 0:
            label = f"{method.replace('_', ' ').title()} ({impl.upper()})"
            color = colors.get((method, impl), None)
            marker = markers.get(impl, 'o')

            ax.plot(
                n_values,
                speedups,
                label=label,
                marker=marker,
                markersize=10,
                color=color,
                linestyle='-',
                linewidth=2.5,
                alpha=0.8
            )

    # Add baseline reference line at y=1
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.7,
               label='Baseline (Direct CuPy)')

    ax.set_xlabel('Number of Particles (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (relative to Direct CuPy)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Factors Relative to Direct Summation CuPy\n(ε=0.01, K=100,000 for MC)',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'speedup_factors.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_mc_error_vs_k(df, output_dir):
    """
    Plot 3: Monte Carlo error vs. K (number of samples)
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Filter for Monte Carlo only
    df_mc = df[df['method'] == 'monte_carlo'].copy()
    df_mc = df_mc[df_mc['softening'] == 0.01]  # Use standard softening

    # Group by implementation and n_bodies
    for impl in df_mc['implementation'].unique():
        df_impl = df_mc[df_mc['implementation'] == impl]

        for n in sorted(df_impl['n_bodies'].unique()):
            df_n = df_impl[df_impl['n_bodies'] == n].copy()
            df_n = df_n.sort_values('n_samples')

            if len(df_n) > 0:
                label = f"N={n:,} ({impl.upper()})"
                marker = 'o' if impl == 'cuda' else 's'

                ax.plot(
                    df_n['n_samples'],
                    df_n['relative_error_pct'],
                    label=label,
                    marker=marker,
                    markersize=8,
                    linestyle='-',
                    linewidth=2,
                    alpha=0.8
                )

    # Add reference line showing 1/sqrt(K) behavior
    k_range = np.logspace(3, 5.5, 100)
    reference = 100 / np.sqrt(k_range)  # Scaled for visibility
    ax.plot(k_range, reference, 'k--', linewidth=1.5, alpha=0.5,
            label=r'Reference: $\propto 1/\sqrt{K}$')

    ax.set_xlabel('Number of Samples (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('Monte Carlo Error vs. Number of Samples\n(ε=0.01)',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mc_error_vs_k.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_comparison(df, output_dir):
    """
    Plot 4: Accuracy comparison across methods as a bar chart
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Filter for standard parameters
    df_plot = df.copy()
    mc_mask = (df_plot['method'] == 'monte_carlo') & (df_plot['n_samples'] == 100000)
    ds_mask = df_plot['method'] == 'direct_summation'
    dl_mask = df_plot['method'] == 'deep_learning'

    df_plot = df_plot[mc_mask | ds_mask | dl_mask]
    df_plot = df_plot[df_plot['softening'] == 0.01]

    # Get unique N values and method/impl combinations
    n_values = sorted(df_plot['n_bodies'].unique())
    methods = df_plot.groupby(['method', 'implementation'])

    # Set up bar positions
    x = np.arange(len(n_values))
    width = 0.15

    colors = {
        ('direct_summation', 'cuda'): '#1f77b4',
        ('direct_summation', 'cupy'): '#ff7f0e',
        ('monte_carlo', 'cuda'): '#2ca02c',
        ('monte_carlo', 'cupy'): '#d62728',
        ('deep_learning', 'pytorch'): '#9467bd',
    }

    i = 0
    for (method, impl), group in methods:
        errors = []
        for n in n_values:
            subset = group[group['n_bodies'] == n]
            if len(subset) > 0:
                errors.append(subset['relative_error_pct'].values[0])
            else:
                errors.append(0)

        label = f"{method.replace('_', ' ').title()} ({impl.upper()})"
        color = colors.get((method, impl), None)

        ax.bar(x + i * width, errors, width, label=label, color=color, alpha=0.8)
        i += 1

    ax.set_xlabel('Number of Particles (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison Across Methods\n(ε=0.01, K=100,000 for MC)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f'{n:,}' for n in n_values])
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Add reference line at 1% error
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='1% threshold (DL target)')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_softening_study(df, output_dir):
    """
    Plot 5: Effect of softening parameter on error
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Filter for Monte Carlo with K=100000
    df_mc = df[(df['method'] == 'monte_carlo') & (df['n_samples'] == 100000)].copy()

    # Plot for different N values
    for n in sorted(df_mc['n_bodies'].unique()):
        df_n = df_mc[df_mc['n_bodies'] == n]

        for impl in df_n['implementation'].unique():
            df_impl = df_n[df_n['implementation'] == impl].copy()
            df_impl = df_impl.sort_values('softening')

            if len(df_impl) > 0:
                label = f"N={n:,} ({impl.upper()})"
                marker = 'o' if impl == 'cuda' else 's'

                # Error plot
                ax1.plot(
                    df_impl['softening'],
                    df_impl['relative_error_pct'],
                    label=label,
                    marker=marker,
                    markersize=8,
                    linestyle='-',
                    linewidth=2,
                    alpha=0.8
                )

    ax1.set_xlabel('Softening Parameter (ε)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Monte Carlo Error vs. Softening Parameter\n(K=100,000)',
                  fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='best', frameon=True, shadow=True, fontsize=8)
    ax1.grid(True, alpha=0.3, which='both')

    # Filter for Direct Summation
    df_ds = df[df['method'] == 'direct_summation'].copy()

    for n in sorted(df_ds['n_bodies'].unique()):
        df_n = df_ds[df_ds['n_bodies'] == n]

        for impl in df_n['implementation'].unique():
            df_impl = df_n[df_n['implementation'] == impl].copy()
            df_impl = df_impl.sort_values('softening')

            if len(df_impl) > 0:
                label = f"N={n:,} ({impl.upper()})"
                marker = 'o' if impl == 'cuda' else 's'

                ax2.plot(
                    df_impl['softening'],
                    df_impl['relative_error_pct'],
                    label=label,
                    marker=marker,
                    markersize=8,
                    linestyle='-',
                    linewidth=2,
                    alpha=0.8
                )

    ax2.set_xlabel('Softening Parameter (ε)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Direct Summation Error vs. Softening Parameter',
                  fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(loc='best', frameon=True, shadow=True, fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'softening_study.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_method_comparison_summary(df, output_dir):
    """
    Plot 6: Summary comparison of all methods (2x2 grid)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Filter data
    df_plot = df.copy()
    mc_mask = (df_plot['method'] == 'monte_carlo') & (df_plot['n_samples'] == 100000)
    ds_mask = df_plot['method'] == 'direct_summation'
    df_plot = df_plot[mc_mask | ds_mask]
    df_plot = df_plot[df_plot['softening'] == 0.01]

    # Colors and markers
    colors = {
        ('direct_summation', 'cuda'): '#1f77b4',
        ('direct_summation', 'cupy'): '#ff7f0e',
        ('monte_carlo', 'cuda'): '#2ca02c',
        ('monte_carlo', 'cupy'): '#d62728',
    }

    markers = {
        'cuda': 'o',
        'cupy': 's',
    }

    # Plot 1: Runtime comparison (linear scale)
    methods = df_plot.groupby(['method', 'implementation'])
    for (method, impl), group in methods:
        group = group.sort_values('n_bodies')
        label = f"{method.replace('_', ' ').title()} ({impl.upper()})"
        color = colors.get((method, impl), None)
        marker = markers.get(impl, 'o')

        ax1.plot(
            group['n_bodies'],
            group['mean_time_s'],
            label=label,
            marker=marker,
            markersize=8,
            color=color,
            linestyle='-',
            linewidth=2,
            alpha=0.8
        )

    ax1.set_xlabel('N', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Runtime (s)', fontsize=11, fontweight='bold')
    ax1.set_title('Runtime vs. N (Linear Scale)', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(loc='best', frameon=True, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error comparison
    for (method, impl), group in methods:
        group = group.sort_values('n_bodies')
        label = f"{method.replace('_', ' ').title()} ({impl.upper()})"
        color = colors.get((method, impl), None)
        marker = markers.get(impl, 'o')

        ax2.plot(
            group['n_bodies'],
            group['relative_error_pct'],
            label=label,
            marker=marker,
            markersize=8,
            color=color,
            linestyle='-',
            linewidth=2,
            alpha=0.8
        )

    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
                label='1% threshold')
    ax2.set_xlabel('N', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Relative Error (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Accuracy vs. N', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(loc='best', frameon=True, fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Runtime efficiency (runtime per particle)
    for (method, impl), group in methods:
        group = group.sort_values('n_bodies')
        runtime_per_particle = group['mean_time_s'] / group['n_bodies']
        label = f"{method.replace('_', ' ').title()} ({impl.upper()})"
        color = colors.get((method, impl), None)
        marker = markers.get(impl, 'o')

        ax3.plot(
            group['n_bodies'],
            runtime_per_particle,
            label=label,
            marker=marker,
            markersize=8,
            color=color,
            linestyle='-',
            linewidth=2,
            alpha=0.8
        )

    ax3.set_xlabel('N', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Runtime per Particle (s)', fontsize=11, fontweight='bold')
    ax3.set_title('Efficiency vs. N', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend(loc='best', frameon=True, fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Accuracy-Speed tradeoff (scatter)
    for (method, impl), group in methods:
        label = f"{method.replace('_', ' ').title()} ({impl.upper()})"
        color = colors.get((method, impl), None)
        marker = markers.get(impl, 'o')

        # Add text labels for N values
        for _, row in group.iterrows():
            ax4.scatter(
                row['mean_time_s'],
                row['relative_error_pct'],
                s=200,
                color=color,
                marker=marker,
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5
            )
            ax4.annotate(
                f"N={row['n_bodies']:,}",
                (row['mean_time_s'], row['relative_error_pct']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=7,
                alpha=0.8
            )

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
               markersize=10, label='Direct CUDA'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e',
               markersize=10, label='Direct CuPy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=10, label='MC CUDA'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728',
               markersize=10, label='MC CuPy'),
    ]

    ax4.set_xlabel('Runtime (s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Relative Error (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Accuracy-Speed Tradeoff', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend(handles=legend_elements, loc='best', frameon=True, fontsize=8)
    ax4.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'method_comparison_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_table(df, output_dir):
    """
    Generate a summary table in text format
    """
    output_path = os.path.join(output_dir, 'summary_table.txt')

    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("GRAVITATIONAL POTENTIAL CALCULATION - RESULTS SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Analytical Reference Value: U = {U_ANALYTICAL}\n\n")

        # Filter for standard parameters
        df_standard = df.copy()
        mc_mask = (df_standard['method'] == 'monte_carlo') & (df_standard['n_samples'] == 100000)
        ds_mask = df_standard['method'] == 'direct_summation'
        df_standard = df_standard[mc_mask | ds_mask]
        df_standard = df_standard[df_standard['softening'] == 0.01]

        f.write("Standard Configuration (ε=0.01, K=100,000 for MC):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Method':<20} {'Impl':<8} {'N':<12} {'Runtime (s)':<15} {'Error (%)':<12} {'U Computed':<15}\n")
        f.write("-" * 100 + "\n")

        for _, row in df_standard.sort_values(['method', 'implementation', 'n_bodies']).iterrows():
            f.write(f"{row['method']:<20} {row['implementation']:<8} {row['n_bodies']:<12,} "
                   f"{row['mean_time_s']:>8.6f} ± {row['std_time_s']:<.2e}  "
                   f"{row['relative_error_pct']:>8.4f}     "
                   f"{row['U_computed']:>10.6f}\n")

        f.write("\n" + "=" * 100 + "\n")

    print(f"Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <path_to_results.csv>")
        print("\nExample:")
        print("  python plot_results.py ./experiments/run_20251127_094351/results.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(csv_path), 'plots')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    # Load data
    df = load_data(csv_path)

    print("\nGenerating plots...")
    print("-" * 50)

    # Generate all plots
    plot_runtime_vs_n(df, output_dir)
    plot_speedup_factors(df, output_dir)
    plot_mc_error_vs_k(df, output_dir)
    plot_accuracy_comparison(df, output_dir)
    plot_softening_study(df, output_dir)
    plot_method_comparison_summary(df, output_dir)

    # Generate summary table
    generate_summary_table(df, output_dir)

    print("-" * 50)
    print(f"\n✓ All plots generated successfully!")
    print(f"✓ Output directory: {output_dir}")
    print(f"\nGenerated files:")
    for filename in sorted(os.listdir(output_dir)):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
