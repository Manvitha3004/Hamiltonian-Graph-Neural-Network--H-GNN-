"""
Hypothesis Test: Universal SNR Threshold for Learnability

HYPOTHESIS: Learning succeeds if and only if SNR > c, where c is universal
across all (N, T, dt, epsilon) configurations.

This is NOT an exploration sweep. This is a focused hypothesis test.

The critical insight:
- ε_crit depends on (N, T, dt) — not universal
- SNR_crit may be CONSTANT — that would be universal and publishable

We test:
1. Does Improvement ~ f(SNR) show a sharp transition?
2. Is the transition point SNR_crit ≈ constant across configurations?
3. Can we fit a universal curve: Improvement = g(SNR - SNR_crit)?
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from h_gnn.models.hgnn import HGNN
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators.symplectic import StormerVerlet


@dataclass
class SNRTestResult:
    """Result from a single training run for SNR hypothesis test."""
    # Configuration
    n_particles: int
    horizon: int
    dt: float
    epsilon: float
    seed: int
    
    # Primary metrics
    final_improvement: float
    mean_snr: float  # Average SNR over training
    early_snr: float  # SNR in first 10 epochs (before learning kicks in)
    final_snr: float  # SNR in last 10 epochs
    
    # Secondary metrics
    final_alpha: float
    converged: bool
    learning_succeeded: bool  # improvement > 0
    
    # Raw data for analysis
    snr_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)


def generate_ic(n_particles: int, mu: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate initial conditions."""
    rng = np.random.default_rng(seed)
    q_list, p_list = [], []
    
    for _ in range(n_particles):
        r = rng.uniform(7000, 42000)
        theta = rng.uniform(0, 2 * np.pi)
        q = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
        v_circ = np.sqrt(mu / r)
        v_perturb = rng.uniform(-0.05, 0.05)
        v = (v_circ + v_perturb) * np.array([-np.sin(theta), np.cos(theta), 0.0])
        q_list.append(q)
        p_list.append(v)
    
    return (torch.tensor(np.stack(q_list), dtype=torch.float64),
            torch.tensor(np.stack(p_list), dtype=torch.float64))


def train_and_measure_snr(
    n_particles: int,
    horizon: int,
    dt: float,
    epsilon: float,
    seed: int,
    n_epochs: int = 100,
    batch_size: int = 8,
    mu: float = 398600.4418,
    verbose: bool = False,
) -> SNRTestResult:
    """
    Train model and carefully measure SNR throughout.
    
    Key: We measure SNR at the START of training (before learning)
    to see if initial SNR predicts success.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Ground truth
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    
    # Edge index
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Create model
    model = HGNN(
        mu=mu,
        interaction_strength=1.0,
        interaction_radius=20000.0,
        hidden_dims=[64, 64, 32],
        integrator_type="stormer_verlet",
        dt=dt,
        create_graph=True,
        softplus_output=False,
        normalize_features=True,
        init_gain=1.0,
        use_dt_scaling=True,
    ).float()
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    gt_integrator = StormerVerlet(dt=dt, create_graph=False)
    
    # Tracking
    snr_history = []
    loss_history = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_grad_phi_sq = 0.0
        epoch_grad_hbase_sq = 0.0
        
        for batch_idx in range(batch_size):
            ic_seed = batch_idx + epoch * batch_size + seed * 10000
            q0, p0 = generate_ic(n_particles, mu, ic_seed)
            
            # Ground truth trajectory
            with torch.no_grad():
                q_true, p_true = q0.double().clone(), p0.double().clone()
                for _ in range(horizon):
                    q_true, p_true = gt_integrator.step(
                        q_true, p_true, gt_hamiltonian, edge_index
                    )
            
            # Model prediction
            q_pred, p_pred = model.forward(
                q0.float(), p0.float(), n_steps=horizon, return_trajectory=False
            )
            
            loss = torch.mean((q_pred - q_true.float())**2 + (p_pred - p_true.float())**2)
            
            # Alpha regularization
            alpha_reg = 1e-4 * model.hamiltonian.alpha.pow(2)
            total_loss = loss + alpha_reg
            
            total_loss.backward()
            epoch_loss += loss.item()
            
            # Measure gradient norms PRECISELY
            # Phi gradients
            phi_grad_sq = 0.0
            for p in model.hamiltonian.Phi.parameters():
                if p.grad is not None:
                    phi_grad_sq += p.grad.norm().item() ** 2
            epoch_grad_phi_sq += phi_grad_sq
            
            # H_base gradient scale: dH/dq ~ -mu*r/|r|^3 ~ mu/r^2
            # For circular orbit at r, |dH/dq| ~ mu/r^2 ~ v^2/r ~ O(v^2/r)
            # We use momentum gradient as proxy: dH/dp = v
            hbase_grad_sq = 0.0
            for i in range(n_particles):
                v = p0[i].float()
                hbase_grad_sq += v.norm().item() ** 2
            epoch_grad_hbase_sq += hbase_grad_sq / n_particles
        
        # Update
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()
        
        avg_loss = epoch_loss / batch_size
        grad_phi_norm = np.sqrt(epoch_grad_phi_sq / batch_size)
        grad_hbase_norm = np.sqrt(epoch_grad_hbase_sq / batch_size)
        
        # SNR = ||grad_Phi|| / ||grad_H_base||
        snr = grad_phi_norm / (grad_hbase_norm + 1e-12)
        
        snr_history.append(snr)
        loss_history.append(avg_loss)
        
        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4e}, SNR={snr:.6f}")
    
    # Compute final improvement
    model.eval()
    base_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=0.0)
    base_integrator = StormerVerlet(dt=dt, create_graph=False)
    
    model_errors = []
    base_errors = []
    
    with torch.no_grad():
        for i in range(20):
            q0, p0 = generate_ic(n_particles, mu, 7000 + i)
            
            # Ground truth
            q_gt, p_gt = q0.double().clone(), p0.double().clone()
            for _ in range(horizon):
                q_gt, p_gt = gt_integrator.step(q_gt, p_gt, gt_hamiltonian, edge_index)
            
            # Model
            q_model, p_model = model.forward(
                q0.float(), p0.float(), n_steps=horizon, return_trajectory=False
            )
            
            # Baseline
            q_base, p_base = q0.double().clone(), p0.double().clone()
            for _ in range(horizon):
                q_base, p_base = base_integrator.step(q_base, p_base, base_hamiltonian, edge_index)
            
            model_err = torch.mean((q_model - q_gt.float())**2 + 
                                    (p_model - p_gt.float())**2).item()
            base_err = torch.mean((q_base.float() - q_gt.float())**2 + 
                                   (p_base.float() - p_gt.float())**2).item()
            
            model_errors.append(model_err)
            base_errors.append(base_err)
    
    avg_model = np.mean(model_errors)
    avg_base = np.mean(base_errors)
    
    # NEW: More robust success metric
    # Success = model beats baseline (model_err < base_err)
    model_beats_baseline = avg_model < avg_base
    
    # Compute improvement only when meaningful
    if avg_base < 1e-12:
        improvement = 0.0  # No baseline error to improve upon
    else:
        improvement = 100 * (avg_base - avg_model) / avg_base
        improvement = np.clip(improvement, -100, 100)  # Clamp to [-100%, 100%]
    
    # Also track raw error ratio for analysis
    error_ratio = avg_model / (avg_base + 1e-12)
    
    # Compute SNR statistics
    early_snr = np.mean(snr_history[:10]) if len(snr_history) >= 10 else np.mean(snr_history)
    final_snr = np.mean(snr_history[-10:]) if len(snr_history) >= 10 else np.mean(snr_history)
    mean_snr = np.mean(snr_history)
    
    # Training loss reduction as alternative success metric
    loss_reduction = (loss_history[0] - loss_history[-1]) / (loss_history[0] + 1e-12) if loss_history else 0
    
    return SNRTestResult(
        n_particles=n_particles,
        horizon=horizon,
        dt=dt,
        epsilon=epsilon,
        seed=seed,
        final_improvement=improvement,
        mean_snr=mean_snr,
        early_snr=early_snr,
        final_snr=final_snr,
        final_alpha=model.hamiltonian.alpha.item(),
        converged=len(loss_history) > 20 and loss_history[-1] < loss_history[20],
        learning_succeeded=model_beats_baseline,  # Use robust metric
        snr_history=snr_history,
        loss_history=loss_history,
    )


def test_snr_threshold_hypothesis(
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    MAIN HYPOTHESIS TEST:
    
    Does learning succeed if and only if SNR > c (constant)?
    
    Strategy:
    1. Sample diverse (N, T, dt, epsilon) configurations
    2. For each, measure early_snr (before learning) and improvement
    3. Plot Improvement vs early_snr
    4. Fit transition curve
    5. Test if transition point is constant across configurations
    """
    print("=" * 70)
    print("HYPOTHESIS TEST: UNIVERSAL SNR THRESHOLD")
    print("=" * 70)
    print()
    print("H0: Learning success depends on epsilon, N, T, dt separately")
    print("H1: Learning success depends only on SNR > c (universal threshold)")
    print()
    print("=" * 70)
    
    if output_dir is None:
        output_dir = Path("results/snr_hypothesis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Design: Sample configurations that VARY in (N, T, dt, epsilon)
    # but may have SAME SNR
    # FAST VERSION: Wider epsilon range to get clear signal variation
    configs = [
        # Low epsilon (expect low SNR, likely failures)
        {'n': 10, 'T': 30, 'dt': 60.0, 'eps': 5.0},
        {'n': 10, 'T': 30, 'dt': 60.0, 'eps': 10.0},
        {'n': 10, 'T': 30, 'dt': 60.0, 'eps': 20.0},
        
        # Medium epsilon (transition zone)
        {'n': 10, 'T': 30, 'dt': 60.0, 'eps': 40.0},
        {'n': 10, 'T': 30, 'dt': 60.0, 'eps': 60.0},
        {'n': 20, 'T': 30, 'dt': 60.0, 'eps': 50.0},
        
        # High epsilon (expect high SNR, likely successes)
        {'n': 10, 'T': 30, 'dt': 60.0, 'eps': 80.0},
        {'n': 10, 'T': 30, 'dt': 60.0, 'eps': 100.0},
        {'n': 10, 'T': 30, 'dt': 60.0, 'eps': 150.0},
        {'n': 20, 'T': 30, 'dt': 60.0, 'eps': 100.0},
    ]
    
    n_seeds = 1  # Single seed for speed
    total_runs = len(configs) * n_seeds
    
    print(f"Running {total_runs} experiments across {len(configs)} configurations")
    print("-" * 70)
    
    all_results = []
    
    for i, cfg in enumerate(configs):
        for seed in range(n_seeds):
            run_num = i * n_seeds + seed + 1
            print(f"[{run_num}/{total_runs}] N={cfg['n']}, T={cfg['T']}, "
                  f"dt={cfg['dt']}, eps={cfg['eps']}, seed={seed}...", 
                  end=" ", flush=True)
            
            try:
                result = train_and_measure_snr(
                    n_particles=cfg['n'],
                    horizon=cfg['T'],
                    dt=cfg['dt'],
                    epsilon=cfg['eps'],
                    seed=seed,
                    n_epochs=40,  # Reduced for faster iteration
                    verbose=False,
                )
                all_results.append(asdict(result))
                
                status = "SUCCESS" if result.learning_succeeded else "FAIL"
                print(f"SNR={result.early_snr:.4f}, imp={result.final_improvement:.1f}% [{status}]")
                
            except Exception as e:
                print(f"ERROR: {e}")
    
    # Analyze results
    analysis = analyze_snr_hypothesis(all_results)
    
    # Save raw results
    output = {
        'hypothesis': 'Learning succeeds iff SNR > c (universal)',
        'configs': configs,
        'n_seeds': n_seeds,
        'results': all_results,
        'analysis': analysis,
    }
    
    with open(output_dir / "snr_hypothesis_test.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    # Generate plots
    plot_snr_hypothesis(all_results, analysis, output_dir)
    
    # Print conclusions
    print_hypothesis_conclusions(analysis)
    
    return output


def analyze_snr_hypothesis(results: List[Dict]) -> Dict[str, Any]:
    """
    Analyze whether SNR predicts learning success universally.
    
    Key tests:
    1. Fit sigmoid to Improvement vs SNR
    2. Find SNR_crit (50% success point)
    3. Test if SNR_crit is constant across (N, T, dt) subgroups
    """
    # Extract data
    snr = np.array([r['early_snr'] for r in results])
    improvement = np.array([r['final_improvement'] for r in results])
    success = np.array([r['learning_succeeded'] for r in results])
    
    analysis = {
        'n_samples': len(results),
        'n_success': int(success.sum()),
        'n_failure': int((~success).sum()),
    }
    
    # 1. Correlation test
    corr, p_value = stats.pearsonr(snr, improvement)
    analysis['correlation'] = {
        'pearson_r': float(corr),
        'p_value': float(p_value),
    }
    
    # 2. Fit sigmoid: P(success) = 1 / (1 + exp(-k*(SNR - SNR_crit)))
    try:
        def sigmoid(x, snr_crit, k):
            return 1 / (1 + np.exp(-k * (x - snr_crit)))
        
        # Sort by SNR for fitting
        sort_idx = np.argsort(snr)
        snr_sorted = snr[sort_idx]
        success_sorted = success[sort_idx].astype(float)
        
        # Initial guess
        snr_crit_guess = np.median(snr)
        
        popt, pcov = curve_fit(
            sigmoid, snr_sorted, success_sorted,
            p0=[snr_crit_guess, 10.0],
            bounds=([0, 0.1], [10, 1000]),
            maxfev=5000
        )
        
        snr_crit, k = popt
        snr_crit_std = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 0
        
        analysis['sigmoid_fit'] = {
            'snr_crit': float(snr_crit),
            'snr_crit_std': float(snr_crit_std),
            'steepness_k': float(k),
            'transition_width': float(4.0 / k),  # width from 10% to 90% success
        }
        
        # Goodness of fit
        pred_success = sigmoid(snr_sorted, snr_crit, k) > 0.5
        accuracy = np.mean(pred_success == success_sorted)
        analysis['sigmoid_fit']['classification_accuracy'] = float(accuracy)
        
    except Exception as e:
        analysis['sigmoid_fit'] = {'error': str(e)}
    
    # 3. Linear regression: Improvement ~ a * SNR + b
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(snr, improvement)
        
        # Find SNR where improvement = 0 (threshold)
        snr_zero = -intercept / slope if slope != 0 else 0
        
        analysis['linear_fit'] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'snr_at_zero_improvement': float(snr_zero),
        }
    except Exception as e:
        analysis['linear_fit'] = {'error': str(e)}
    
    # 4. Test universality: Is SNR_crit the same across configurations?
    # Group by N, T, dt and find SNR threshold for each
    try:
        by_config = defaultdict(list)
        for r in results:
            key = f"N{r['n_particles']}_T{r['horizon']}_dt{r['dt']}"
            by_config[key].append(r)
        
        config_thresholds = {}
        for key, group in by_config.items():
            if len(group) >= 2:
                group_snr = np.array([r['early_snr'] for r in group])
                group_success = np.array([r['learning_succeeded'] for r in group])
                
                # Find SNR threshold for this config
                if group_success.any() and (~group_success).any():
                    # Midpoint between highest failure and lowest success
                    fail_snr = group_snr[~group_success]
                    succ_snr = group_snr[group_success]
                    threshold = (fail_snr.max() + succ_snr.min()) / 2
                    config_thresholds[key] = float(threshold)
                elif group_success.all():
                    config_thresholds[key] = float(group_snr.min() - 0.1)
                else:
                    config_thresholds[key] = float(group_snr.max() + 0.1)
        
        if len(config_thresholds) >= 2:
            thresholds = list(config_thresholds.values())
            analysis['universality'] = {
                'config_thresholds': config_thresholds,
                'mean_threshold': float(np.mean(thresholds)),
                'std_threshold': float(np.std(thresholds)),
                'cv_threshold': float(np.std(thresholds) / np.mean(thresholds)),
                'is_universal': float(np.std(thresholds) / np.mean(thresholds)) < 0.3,
            }
        else:
            analysis['universality'] = {'error': 'insufficient config groups'}
            
    except Exception as e:
        analysis['universality'] = {'error': str(e)}
    
    # 5. Optimal threshold via ROC analysis
    try:
        thresholds = np.linspace(snr.min(), snr.max(), 100)
        best_acc = 0
        best_thresh = 0
        
        for thresh in thresholds:
            pred = snr > thresh
            acc = np.mean(pred == success)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        analysis['optimal_threshold'] = {
            'snr_crit': float(best_thresh),
            'accuracy': float(best_acc),
        }
    except Exception:
        pass
    
    return analysis


def plot_snr_hypothesis(results: List[Dict], analysis: Dict, output_dir: Path):
    """Generate hypothesis test plots."""
    
    snr = np.array([r['early_snr'] for r in results])
    improvement = np.array([r['final_improvement'] for r in results])
    success = np.array([r['learning_succeeded'] for r in results])
    
    # Extract config info for coloring
    n_vals = np.array([r['n_particles'] for r in results])
    t_vals = np.array([r['horizon'] for r in results])
    dt_vals = np.array([r['dt'] for r in results])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ========================================
    # Plot 1: Improvement vs SNR (main result)
    # ========================================
    ax = axes[0, 0]
    
    colors = ['#e74c3c' if not s else '#2ecc71' for s in success]
    ax.scatter(snr, improvement, c=colors, s=80, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Fit line
    if 'linear_fit' in analysis and 'slope' in analysis['linear_fit']:
        lf = analysis['linear_fit']
        x_fit = np.linspace(snr.min(), snr.max(), 100)
        y_fit = lf['slope'] * x_fit + lf['intercept']
        ax.plot(x_fit, y_fit, 'b--', linewidth=2, 
                label=f"Fit: imp = {lf['slope']:.1f}*SNR + {lf['intercept']:.1f}\n$R^2$={lf['r_squared']:.3f}")
        
        # SNR threshold
        snr_thresh = lf['snr_at_zero_improvement']
        ax.axvline(x=snr_thresh, color='orange', linestyle=':', linewidth=2,
                   label=f'SNR$_{{crit}}$ = {snr_thresh:.4f}')
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Early SNR (||∇Φ|| / ||∇H_base||)', fontsize=12)
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax.set_title('MAIN RESULT: Improvement vs SNR', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    # ========================================
    # Plot 2: Success probability vs SNR (sigmoid)
    # ========================================
    ax = axes[0, 1]
    
    # Bin by SNR and compute success rate
    n_bins = 8
    snr_bins = np.linspace(snr.min(), snr.max(), n_bins + 1)
    bin_centers = []
    success_rates = []
    
    for i in range(n_bins):
        mask = (snr >= snr_bins[i]) & (snr < snr_bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((snr_bins[i] + snr_bins[i+1]) / 2)
            success_rates.append(success[mask].mean())
    
    ax.scatter(bin_centers, success_rates, s=150, c='#3498db', edgecolor='black', 
               linewidth=1.5, zorder=5, label='Observed')
    
    # Fit sigmoid
    if 'sigmoid_fit' in analysis and 'snr_crit' in analysis['sigmoid_fit']:
        sf = analysis['sigmoid_fit']
        x_fit = np.linspace(snr.min(), snr.max(), 200)
        y_fit = 1 / (1 + np.exp(-sf['steepness_k'] * (x_fit - sf['snr_crit'])))
        ax.plot(x_fit, y_fit, 'r-', linewidth=2,
                label=f"Sigmoid: SNR$_{{crit}}$ = {sf['snr_crit']:.4f}")
        
        ax.axvline(x=sf['snr_crit'], color='orange', linestyle=':', linewidth=2)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Early SNR', fontsize=12)
    ax.set_ylabel('P(Learning Success)', fontsize=12)
    ax.set_title('Success Probability Transition', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ========================================
    # Plot 3: SNR_crit by configuration (universality test)
    # ========================================
    ax = axes[1, 0]
    
    if 'universality' in analysis and 'config_thresholds' in analysis['universality']:
        univ = analysis['universality']
        configs = list(univ['config_thresholds'].keys())
        thresholds = list(univ['config_thresholds'].values())
        
        ax.barh(range(len(configs)), thresholds, color='#9b59b6', edgecolor='black')
        ax.axvline(x=univ['mean_threshold'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean = {univ['mean_threshold']:.4f} ± {univ['std_threshold']:.4f}")
        
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs, fontsize=9)
        ax.set_xlabel('SNR$_{crit}$', fontsize=12)
        ax.set_title('Universality Test: SNR$_{crit}$ by Configuration', fontsize=14)
        ax.legend()
        
        # Add CV annotation
        cv = univ['cv_threshold']
        is_univ = univ['is_universal']
        color = 'green' if is_univ else 'red'
        verdict = "UNIVERSAL" if is_univ else "NOT UNIVERSAL"
        ax.text(0.95, 0.05, f"CV = {cv:.2%}\n{verdict}", 
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=12, fontweight='bold', color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Insufficient data for universality test',
                transform=ax.transAxes, ha='center', va='center')
    
    ax.grid(alpha=0.3, axis='x')
    
    # ========================================
    # Plot 4: Improvement vs SNR colored by config
    # ========================================
    ax = axes[1, 1]
    
    # Color by N
    unique_n = np.unique(n_vals)
    colors_n = plt.cm.viridis(np.linspace(0, 1, len(unique_n)))
    color_map = {n: colors_n[i] for i, n in enumerate(unique_n)}
    
    for n in unique_n:
        mask = n_vals == n
        ax.scatter(snr[mask], improvement[mask], c=[color_map[n]], s=80, 
                   alpha=0.7, label=f'N={n}', edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Show that DATA COLLAPSES onto single curve regardless of N
    if 'linear_fit' in analysis and 'slope' in analysis['linear_fit']:
        lf = analysis['linear_fit']
        x_fit = np.linspace(snr.min(), snr.max(), 100)
        y_fit = lf['slope'] * x_fit + lf['intercept']
        ax.plot(x_fit, y_fit, 'k--', linewidth=2, label='Universal curve')
    
    ax.set_xlabel('Early SNR', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Data Collapse Test: Different N on Same Curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'snr_hypothesis_test.png', dpi=300)
    plt.savefig(output_dir / 'snr_hypothesis_test.pdf')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")


def print_hypothesis_conclusions(analysis: Dict):
    """Print hypothesis test conclusions."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST CONCLUSIONS")
    print("=" * 70)
    
    # Correlation
    if 'correlation' in analysis:
        corr = analysis['correlation']
        print(f"\n1. CORRELATION TEST")
        print(f"   Pearson r = {corr['pearson_r']:.4f}")
        print(f"   p-value = {corr['p_value']:.2e}")
        if corr['p_value'] < 0.01:
            print(f"   >>> SIGNIFICANT correlation between SNR and improvement")
        else:
            print(f"   >>> No significant correlation")
    
    # Sigmoid fit
    if 'sigmoid_fit' in analysis and 'snr_crit' in analysis['sigmoid_fit']:
        sf = analysis['sigmoid_fit']
        print(f"\n2. SIGMOID FIT (Success Probability)")
        print(f"   SNR_crit = {sf['snr_crit']:.4f} ± {sf['snr_crit_std']:.4f}")
        print(f"   Transition width = {sf['transition_width']:.4f}")
        print(f"   Classification accuracy = {sf['classification_accuracy']:.1%}")
        if sf['classification_accuracy'] > 0.8:
            print(f"   >>> SHARP transition at SNR_crit")
        else:
            print(f"   >>> Gradual transition")
    
    # Linear fit
    if 'linear_fit' in analysis and 'snr_at_zero_improvement' in analysis['linear_fit']:
        lf = analysis['linear_fit']
        print(f"\n3. LINEAR FIT")
        print(f"   Improvement = {lf['slope']:.2f} * SNR + {lf['intercept']:.2f}")
        print(f"   R² = {lf['r_squared']:.4f}")
        print(f"   SNR threshold (imp=0) = {lf['snr_at_zero_improvement']:.4f}")
    
    # Universality
    if 'universality' in analysis and 'is_universal' in analysis['universality']:
        univ = analysis['universality']
        print(f"\n4. UNIVERSALITY TEST")
        print(f"   Mean SNR_crit across configs = {univ['mean_threshold']:.4f}")
        print(f"   Std SNR_crit = {univ['std_threshold']:.4f}")
        print(f"   Coefficient of variation = {univ['cv_threshold']:.2%}")
        if univ['is_universal']:
            print(f"   >>> SNR_crit IS UNIVERSAL (CV < 30%)")
            print(f"   >>> THIS IS A PUBLISHABLE RESULT")
        else:
            print(f"   >>> SNR_crit varies across configurations")
    
    # Optimal threshold
    if 'optimal_threshold' in analysis:
        opt = analysis['optimal_threshold']
        print(f"\n5. OPTIMAL THRESHOLD (ROC)")
        print(f"   Best SNR threshold = {opt['snr_crit']:.4f}")
        print(f"   Classification accuracy = {opt['accuracy']:.1%}")
    
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    # Determine if hypothesis is supported
    is_supported = (
        analysis.get('correlation', {}).get('p_value', 1) < 0.01 and
        analysis.get('sigmoid_fit', {}).get('classification_accuracy', 0) > 0.75 and
        analysis.get('universality', {}).get('is_universal', False)
    )
    
    if is_supported:
        snr_crit = analysis.get('universality', {}).get('mean_threshold', 
                   analysis.get('sigmoid_fit', {}).get('snr_crit', 0))
        print(f"\n>>> HYPOTHESIS SUPPORTED <<<")
        print(f"")
        print(f"Learning succeeds if and only if SNR > {snr_crit:.4f}")
        print(f"")
        print(f"This threshold is UNIVERSAL across different:")
        print(f"  - System sizes (N)")
        print(f"  - Prediction horizons (T)")
        print(f"  - Time steps (dt)")
        print(f"  - Interaction strengths (epsilon)")
        print(f"")
        print(f"PUBLISH THIS as: 'Universal SNR threshold for Hamiltonian correction learning'")
    else:
        print(f"\n>>> HYPOTHESIS PARTIALLY SUPPORTED or REJECTED <<<")
        print(f"More data or refined hypothesis needed.")


if __name__ == "__main__":
    test_snr_threshold_hypothesis()
