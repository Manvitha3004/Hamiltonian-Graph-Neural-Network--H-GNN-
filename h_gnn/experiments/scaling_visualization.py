"""
Visualization tools for scaling law discovery.

Generates publication-quality plots for:
- Epsilon threshold curves
- N-scaling analysis
- Lyapunov exponent comparisons
- dt^2 validation
- Failure mode boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, Any


def set_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (7, 5),
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_epsilon_threshold(results: Dict, output_dir: Path):
    """Plot learnability threshold vs epsilon."""
    set_style()
    
    scaling = results.get('scaling_law', {})
    metrics = results.get('metrics', [])
    
    # Group by epsilon
    by_eps = {}
    for m in metrics:
        eps = m['epsilon']
        if eps not in by_eps:
            by_eps[eps] = []
        by_eps[eps].append(m['final_improvement'])
    
    # Compute means and stds
    epsilons = sorted(by_eps.keys())
    means = [np.mean(by_eps[e]) for e in epsilons]
    stds = [np.std(by_eps[e]) for e in epsilons]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Improvement vs epsilon
    ax1.errorbar(epsilons, means, yerr=stds, fmt='o-', capsize=4, 
                 color='#2ecc71', linewidth=2, markersize=8, label='Mean ± Std')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                label='Zero improvement')
    
    # Critical threshold
    eps_crit = scaling.get('critical_value', 0)
    if eps_crit > 0:
        ax1.axvline(x=eps_crit, color='orange', linestyle=':', linewidth=2,
                    label=f'$\\epsilon_{{crit}} = {eps_crit:.1f}$')
        ax1.fill_betweenx([-20, max(means) + 10], 0, eps_crit, alpha=0.15, 
                          color='red', label='Unlearnable region')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Interaction Strength ($\\epsilon$)')
    ax1.set_ylabel('T=50 Improvement (%)')
    ax1.set_title('Learnability Threshold')
    ax1.legend(loc='best')
    ax1.set_ylim(-20, max(means) + max(stds) + 5)
    
    # Right: Log-log scaling fit
    positive_mask = np.array(means) > 0
    if positive_mask.sum() >= 3:
        eps_pos = np.array(epsilons)[positive_mask]
        imp_pos = np.array(means)[positive_mask]
        
        ax2.loglog(eps_pos, imp_pos, 'o', color='#3498db', markersize=10, label='Data')
        
        # Fit line
        gamma = scaling.get('scaling_exponent', 0)
        c = scaling.get('scaling_coefficient', 1)
        r2 = scaling.get('r_squared', 0)
        
        eps_fit = np.logspace(np.log10(eps_pos.min()), np.log10(eps_pos.max()), 100)
        imp_fit = c * eps_fit ** gamma
        ax2.loglog(eps_fit, imp_fit, '--', color='#e74c3c', linewidth=2,
                   label=f'Fit: $\\propto \\epsilon^{{{gamma:.2f}}}$ ($R^2={r2:.3f}$)')
    
    ax2.set_xlabel('Interaction Strength ($\\epsilon$)')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Scaling Law Fit')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'epsilon_threshold.png', dpi=300)
    plt.savefig(output_dir / 'epsilon_threshold.pdf')
    plt.close()
    print("Saved: epsilon_threshold.png/pdf")


def plot_n_scaling(results: Dict, output_dir: Path):
    """Plot system size scaling analysis."""
    set_style()
    
    scaling = results.get('scaling_law', {})
    metrics = results.get('metrics', [])
    
    # Group by N
    by_n = {}
    for m in metrics:
        n = m['n_particles']
        if n not in by_n:
            by_n[n] = {'improvements': [], 'alphas': []}
        by_n[n]['improvements'].append(m['final_improvement'])
        by_n[n]['alphas'].append(abs(m['final_alpha']))
    
    n_values = sorted(by_n.keys())
    imp_means = [np.mean(by_n[n]['improvements']) for n in n_values]
    imp_stds = [np.std(by_n[n]['improvements']) for n in n_values]
    alpha_means = [np.mean(by_n[n]['alphas']) for n in n_values]
    alpha_stds = [np.std(by_n[n]['alphas']) for n in n_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Improvement vs N
    ax1.errorbar(n_values, imp_means, yerr=imp_stds, fmt='s-', capsize=4,
                 color='#3498db', linewidth=2, markersize=10)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('System Size (N)')
    ax1.set_ylabel('T=50 Improvement (%)')
    ax1.set_title('Improvement vs System Size')
    
    n_crit = scaling.get('critical_value', float('inf'))
    if n_crit < float('inf') and n_crit > 0:
        ax1.axvline(x=n_crit, color='orange', linestyle=':', linewidth=2,
                    label=f'$N_{{crit}} = {n_crit:.0f}$')
        ax1.legend()
    
    # Right: Alpha scaling
    ax2.errorbar(n_values, alpha_means, yerr=alpha_stds, fmt='o-', capsize=4,
                 color='#2ecc71', linewidth=2, markersize=8, label='Mean ± Std')
    ax2.set_xlabel('System Size (N)')
    ax2.set_ylabel('Final $|\\alpha|$')
    ax2.set_title('Correction Magnitude vs System Size')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Fit line
    gamma = scaling.get('scaling_exponent', 0)
    if abs(gamma) > 0:
        n_fit = np.logspace(np.log10(min(n_values)), np.log10(max(n_values)), 100)
        alpha_fit_coeff = alpha_means[0] / (n_values[0] ** gamma)
        alpha_fit = alpha_fit_coeff * n_fit ** gamma
        ax2.loglog(n_fit, alpha_fit, '--', color='#e74c3c', linewidth=2,
                   label=f'$\\alpha \\propto N^{{{gamma:.2f}}}$')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'n_scaling.png', dpi=300)
    plt.savefig(output_dir / 'n_scaling.pdf')
    plt.close()
    print("Saved: n_scaling.png/pdf")


def plot_lyapunov_comparison(results: Dict, output_dir: Path):
    """Plot Lyapunov exponent comparison."""
    set_style()
    
    models = ['physics_only', 'ground_truth', 'hgnn']
    labels = ['Physics-Only', 'Ground Truth', 'H-GNN']
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    lyapunov_means = []
    lyapunov_stds = []
    
    for model in models:
        if model in results:
            lyapunov_means.append(results[model].get('lyapunov_mean', 0))
            lyapunov_stds.append(results[model].get('lyapunov_std', 0))
        else:
            lyapunov_means.append(0)
            lyapunov_stds.append(0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(models))
    bars = ax.bar(x, lyapunov_means, yerr=lyapunov_stds, capsize=6,
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Lyapunov Exponent (1/s)')
    ax.set_title('Chaos Analysis: Lyapunov Exponent Comparison')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, lyapunov_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + lyapunov_stds[i] + 0.00001,
                f'{val:.6f}', ha='center', va='bottom', fontsize=10)
    
    # Add interpretation
    if lyapunov_means[2] < lyapunov_means[0]:  # H-GNN < Physics-only
        ax.annotate('H-GNN reduces\nchaotic divergence',
                    xy=(2, lyapunov_means[2]), xytext=(2.5, lyapunov_means[0]),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lyapunov_comparison.png', dpi=300)
    plt.savefig(output_dir / 'lyapunov_comparison.pdf')
    plt.close()
    print("Saved: lyapunov_comparison.png/pdf")


def plot_dt_scaling(results: Dict, output_dir: Path):
    """Plot backward error dt^2 validation."""
    set_style()
    
    scaling = results.get('scaling_law', {})
    metrics = results.get('metrics', [])
    
    # Group by dt
    by_dt = {}
    for m in metrics:
        dt = m['dt']
        if dt not in by_dt:
            by_dt[dt] = []
        by_dt[dt].append(abs(m['final_alpha']))
    
    dt_values = sorted(by_dt.keys())
    alpha_means = [np.mean(by_dt[dt]) for dt in dt_values]
    alpha_stds = [np.std(by_dt[dt]) for dt in dt_values]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(dt_values, alpha_means, yerr=alpha_stds, fmt='o', capsize=4,
                color='#3498db', markersize=12, linewidth=2, label='Observed')
    
    # Fit line
    gamma = scaling.get('scaling_exponent', 2.0)
    r2 = scaling.get('r_squared', 0)
    
    dt_fit = np.linspace(min(dt_values) * 0.8, max(dt_values) * 1.2, 100)
    alpha_fit_coeff = alpha_means[0] / (dt_values[0] ** gamma)
    alpha_fit = alpha_fit_coeff * dt_fit ** gamma
    
    ax.plot(dt_fit, alpha_fit, '--', color='#e74c3c', linewidth=2,
            label=f'Fit: $\\alpha \\propto dt^{{{gamma:.2f}}}$ ($R^2={r2:.3f}$)')
    
    # Reference dt^2 line
    dt_ref = np.linspace(min(dt_values), max(dt_values), 100)
    ref_coeff = alpha_means[0] / (dt_values[0] ** 2)
    alpha_ref = ref_coeff * dt_ref ** 2
    ax.plot(dt_ref, alpha_ref, ':', color='gray', linewidth=1.5, alpha=0.7,
            label='Reference: $\\alpha \\propto dt^2$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time Step dt (s)')
    ax.set_ylabel('Learned $|\\alpha|$')
    ax.set_title('Backward Error Alignment: dt Scaling')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dt_scaling.png', dpi=300)
    plt.savefig(output_dir / 'dt_scaling.pdf')
    plt.close()
    print("Saved: dt_scaling.png/pdf")


def plot_failure_modes(results: Dict, output_dir: Path):
    """Plot failure mode characterization."""
    set_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Phi collapse (top-left)
    ax = axes[0, 0]
    phi_data = results.get('phi_collapse', [])
    if phi_data:
        epsilons = [d['epsilon'] for d in phi_data]
        collapsed = [d['phi_collapsed'] for d in phi_data]
        improvements = [d['improvement'] for d in phi_data]
        
        colors = ['red' if c else 'green' for c in collapsed]
        ax.scatter(epsilons, improvements, c=colors, s=100, zorder=5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('$\\epsilon$')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Phi Collapse Analysis')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Phi OK'),
                           Patch(facecolor='red', label='Phi Collapsed')]
        ax.legend(handles=legend_elements)
    
    # 2. Alpha divergence (top-right)
    ax = axes[0, 1]
    alpha_data = results.get('alpha_divergence', [])
    if alpha_data:
        lrs = [d['lr'] for d in alpha_data]
        diverged = [d['alpha_diverged'] for d in alpha_data]
        alphas = [abs(d['final_alpha']) for d in alpha_data]
        
        colors = ['red' if d else 'green' for d in diverged]
        ax.scatter(lrs, alphas, c=colors, s=100, zorder=5)
        ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=1.5,
                   label='Divergence threshold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Final $|\\alpha|$')
        ax.set_title('Alpha Divergence Analysis')
        ax.legend()
    
    # 3. N-scaling failure (bottom-left)
    ax = axes[1, 0]
    n_data = results.get('n_scaling_failure', [])
    if n_data:
        ns = [d['n'] for d in n_data]
        failed = [d['training_failed'] for d in n_data]
        improvements = [d['improvement'] for d in n_data]
        
        colors = ['red' if f else 'green' for f in failed]
        ax.bar(range(len(ns)), improvements, color=colors, edgecolor='black')
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([str(n) for n in ns])
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('System Size N')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('N-Scaling Failure Analysis')
    
    # 4. Epsilon failure boundary (bottom-right)
    ax = axes[1, 1]
    eps_data = results.get('epsilon_failure', [])
    if eps_data:
        epsilons = [d['epsilon'] for d in eps_data]
        failed = [d['training_failed'] for d in eps_data]
        snrs = [d.get('gradient_snr', 0) for d in eps_data]
        
        colors = ['red' if f else 'green' for f in failed]
        ax.scatter(epsilons, snrs, c=colors, s=100, zorder=5)
        ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=1.5,
                   label='SNR = 1 threshold')
        ax.set_xscale('log')
        ax.set_xlabel('$\\epsilon$')
        ax.set_ylabel('Gradient SNR')
        ax.set_title('Epsilon Failure Boundary')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'failure_modes.png', dpi=300)
    plt.savefig(output_dir / 'failure_modes.pdf')
    plt.close()
    print("Saved: failure_modes.png/pdf")


def plot_gradient_evolution(metrics_list: list, output_dir: Path):
    """Plot gradient SNR evolution during training."""
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for m in metrics_list[:5]:  # Plot first 5
        if m.get('gradient_snr'):
            epochs = range(len(m['gradient_snr']))
            ax1.semilogy(epochs, m['gradient_snr'], alpha=0.7,
                         label=f"eps={m['epsilon']:.0f}")
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Gradient SNR')
    ax1.set_title('Gradient Signal-to-Noise Ratio Evolution')
    ax1.axhline(y=1.0, color='red', linestyle='--', label='SNR = 1')
    ax1.legend()
    
    # Alpha evolution
    for m in metrics_list[:5]:
        if m.get('alpha_history'):
            epochs = range(len(m['alpha_history']))
            ax2.plot(epochs, m['alpha_history'], alpha=0.7,
                     label=f"eps={m['epsilon']:.0f}")
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('$\\alpha$')
    ax2.set_title('Alpha Evolution During Training')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_evolution.png', dpi=300)
    plt.savefig(output_dir / 'gradient_evolution.pdf')
    plt.close()
    print("Saved: gradient_evolution.png/pdf")


def generate_all_plots(results_dir: Path = None, output_dir: Path = None):
    """Generate all plots from saved results."""
    if results_dir is None:
        results_dir = Path("results/scaling_laws")
    if output_dir is None:
        output_dir = results_dir / "figures"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING SCALING LAW PLOTS")
    print("=" * 70)
    
    # Load and plot each result file
    result_files = {
        'epsilon_threshold.json': plot_epsilon_threshold,
        'n_scaling.json': plot_n_scaling,
        'lyapunov_analysis.json': plot_lyapunov_comparison,
        'dt_scaling.json': plot_dt_scaling,
        'failure_modes.json': plot_failure_modes,
    }
    
    for filename, plot_func in result_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            print(f"\nPlotting {filename}...")
            try:
                with open(filepath) as f:
                    data = json.load(f)
                plot_func(data, output_dir)
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  {filename} not found, skipping")
    
    print("\n" + "=" * 70)
    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    generate_all_plots()
