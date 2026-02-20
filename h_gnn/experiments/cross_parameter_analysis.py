"""
Cross-Parameter Scaling Analysis for H-GNN.

Discovers the combined scaling law:
    ε_crit(N, T, dt) = f(N, T, dt)

by running experiments across the full parameter space and fitting
regression models to identify invariant patterns.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time
from itertools import product
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from h_gnn.experiments.scaling_law_discovery import (
    train_with_logging,
    TrainingMetrics,
)


@dataclass
class CrossParameterResult:
    """Result from cross-parameter sweep."""
    n_particles: int
    horizon: int
    dt: float
    epsilon: float
    seed: int
    
    improvement: float = 0.0
    alpha: float = 0.0
    converged: bool = False
    failed: bool = False
    
    # Derived metrics
    gradient_snr_mean: float = 0.0
    convergence_epoch: int = -1


def cross_parameter_sweep(
    n_values: List[int] = [5, 10, 20],
    t_values: List[int] = [20, 50, 100],
    dt_values: List[float] = [30.0, 60.0, 120.0],
    epsilon_values: List[float] = [1.0, 5.0, 25.0, 100.0],
    n_seeds: int = 2,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Run cross-parameter sweep to discover ε_crit(N, T, dt).
    
    Tests all combinations of (N, T, dt, ε) with multiple seeds.
    """
    print("=" * 70)
    print("CROSS-PARAMETER SCALING SWEEP")
    print("=" * 70)
    
    if output_dir is None:
        output_dir = Path("results/scaling_laws")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_configs = len(n_values) * len(t_values) * len(dt_values) * len(epsilon_values) * n_seeds
    print(f"Total configurations: {total_configs}")
    print(f"N: {n_values}")
    print(f"T: {t_values}")
    print(f"dt: {dt_values}")
    print(f"epsilon: {epsilon_values}")
    print("-" * 70)
    
    all_results = []
    config_idx = 0
    
    for n, t, dt, eps in product(n_values, t_values, dt_values, epsilon_values):
        for seed in range(n_seeds):
            config_idx += 1
            print(f"[{config_idx}/{total_configs}] N={n}, T={t}, dt={dt}, eps={eps}, seed={seed}...", 
                  end=" ", flush=True)
            
            config = {
                'seed': seed,
                'epsilon': eps,
                'n_particles': n,
                'dt': dt,
                'horizon': t,
                'n_epochs': 80,
                'batch_size': 8,
            }
            
            try:
                _, metrics = train_with_logging(config, verbose=False)
                
                result = CrossParameterResult(
                    n_particles=n,
                    horizon=t,
                    dt=dt,
                    epsilon=eps,
                    seed=seed,
                    improvement=metrics.final_improvement,
                    alpha=metrics.final_alpha,
                    converged=metrics.convergence_epoch > 0,
                    failed=metrics.training_failed,
                    gradient_snr_mean=np.mean(metrics.gradient_snr) if metrics.gradient_snr else 0,
                    convergence_epoch=metrics.convergence_epoch,
                )
                all_results.append(asdict(result))
                print(f"imp={metrics.final_improvement:.1f}%")
                
            except Exception as e:
                print(f"ERROR: {e}")
                result = CrossParameterResult(
                    n_particles=n, horizon=t, dt=dt, epsilon=eps, seed=seed,
                    failed=True
                )
                all_results.append(asdict(result))
    
    # Analyze cross-parameter scaling
    scaling_law = fit_cross_parameter_scaling(all_results)
    
    # Save results
    output = {
        'parameters': {
            'n_values': n_values,
            't_values': t_values,
            'dt_values': dt_values,
            'epsilon_values': epsilon_values,
            'n_seeds': n_seeds,
        },
        'results': all_results,
        'scaling_law': scaling_law,
    }
    
    with open(output_dir / "cross_parameter_sweep.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 70)
    print("CROSS-PARAMETER SCALING LAW")
    print("=" * 70)
    print(f"Fitted model: eps_crit ~ N^{scaling_law['n_exponent']:.3f} * T^{scaling_law['t_exponent']:.3f} * dt^{scaling_law['dt_exponent']:.3f}")
    print(f"R-squared (improvement): {scaling_law['r_squared_improvement']:.4f}")
    print(f"R-squared (eps_crit): {scaling_law['r_squared_eps_crit']:.4f}")
    
    return output


def fit_cross_parameter_scaling(results: List[Dict]) -> Dict[str, Any]:
    """
    Fit cross-parameter scaling law to results.
    
    Model:
        improvement ~ eps^a * N^b * T^c * dt^d
        
    And find eps_crit for each (N, T, dt) combination.
    """
    # Extract data
    valid_results = [r for r in results if not r['failed']]
    
    eps = np.array([r['epsilon'] for r in valid_results])
    n = np.array([r['n_particles'] for r in valid_results])
    t = np.array([r['horizon'] for r in valid_results])
    dt = np.array([r['dt'] for r in valid_results])
    imp = np.array([r['improvement'] for r in valid_results])
    alpha = np.array([abs(r['alpha']) for r in valid_results])
    
    scaling_law = {
        'n_exponent': 0.0,
        't_exponent': 0.0,
        'dt_exponent': 0.0,
        'eps_exponent': 0.0,
        'r_squared_improvement': 0.0,
        'r_squared_eps_crit': 0.0,
        'eps_crit_by_config': {},
    }
    
    # 1. Fit log-linear model for improvement
    # log(|imp|) ~ a*log(eps) + b*log(N) + c*log(T) + d*log(dt)
    try:
        positive_imp = imp > 0
        if positive_imp.sum() >= 10:
            log_eps = np.log(eps[positive_imp])
            log_n = np.log(n[positive_imp])
            log_t = np.log(t[positive_imp])
            log_dt = np.log(dt[positive_imp])
            log_imp = np.log(imp[positive_imp])
            
            # Design matrix
            X = np.column_stack([log_eps, log_n, log_t, log_dt, np.ones_like(log_eps)])
            
            # Least squares fit
            coeffs, residuals, rank, s = np.linalg.lstsq(X, log_imp, rcond=None)
            
            scaling_law['eps_exponent'] = coeffs[0]
            scaling_law['n_exponent'] = coeffs[1]
            scaling_law['t_exponent'] = coeffs[2]
            scaling_law['dt_exponent'] = coeffs[3]
            
            # R-squared
            y_pred = X @ coeffs
            ss_res = np.sum((log_imp - y_pred) ** 2)
            ss_tot = np.sum((log_imp - np.mean(log_imp)) ** 2)
            scaling_law['r_squared_improvement'] = 1 - ss_res / (ss_tot + 1e-10)
    except Exception as e:
        print(f"Warning: Could not fit improvement scaling: {e}")
    
    # 2. Find eps_crit for each (N, T, dt) configuration
    try:
        from collections import defaultdict
        by_config = defaultdict(list)
        
        for r in valid_results:
            key = (r['n_particles'], r['horizon'], r['dt'])
            by_config[key].append((r['epsilon'], r['improvement']))
        
        for key, data in by_config.items():
            data.sort(key=lambda x: x[0])  # Sort by epsilon
            eps_vals = [d[0] for d in data]
            imp_vals = [d[1] for d in data]
            
            # Find transition point
            eps_crit = find_transition_epsilon(eps_vals, imp_vals)
            config_str = f"N{key[0]}_T{key[1]}_dt{key[2]}"
            scaling_law['eps_crit_by_config'][config_str] = eps_crit
        
        # Fit eps_crit scaling
        if len(scaling_law['eps_crit_by_config']) >= 4:
            configs = []
            eps_crits = []
            for config_str, eps_crit in scaling_law['eps_crit_by_config'].items():
                if eps_crit > 0 and eps_crit < float('inf'):
                    parts = config_str.split('_')
                    n_val = int(parts[0][1:])
                    t_val = int(parts[1][1:])
                    dt_val = float(parts[2][2:])
                    configs.append((n_val, t_val, dt_val))
                    eps_crits.append(eps_crit)
            
            if len(eps_crits) >= 4:
                configs = np.array(configs)
                eps_crits = np.array(eps_crits)
                
                log_n = np.log(configs[:, 0])
                log_t = np.log(configs[:, 1])
                log_dt = np.log(configs[:, 2])
                log_eps_crit = np.log(eps_crits)
                
                X = np.column_stack([log_n, log_t, log_dt, np.ones_like(log_n)])
                coeffs, _, _, _ = np.linalg.lstsq(X, log_eps_crit, rcond=None)
                
                # These are the scaling exponents for eps_crit ~ N^a * T^b * dt^c
                scaling_law['eps_crit_n_exponent'] = coeffs[0]
                scaling_law['eps_crit_t_exponent'] = coeffs[1]
                scaling_law['eps_crit_dt_exponent'] = coeffs[2]
                
                y_pred = X @ coeffs
                ss_res = np.sum((log_eps_crit - y_pred) ** 2)
                ss_tot = np.sum((log_eps_crit - np.mean(log_eps_crit)) ** 2)
                scaling_law['r_squared_eps_crit'] = 1 - ss_res / (ss_tot + 1e-10)
                
    except Exception as e:
        print(f"Warning: Could not fit eps_crit scaling: {e}")
    
    return scaling_law


def find_transition_epsilon(eps_vals: List[float], imp_vals: List[float]) -> float:
    """Find epsilon where improvement transitions from negative to positive."""
    eps_arr = np.array(eps_vals)
    imp_arr = np.array(imp_vals)
    
    # Average over seeds if duplicates exist
    unique_eps = np.unique(eps_arr)
    avg_imp = []
    for e in unique_eps:
        mask = eps_arr == e
        avg_imp.append(np.mean(imp_arr[mask]))
    avg_imp = np.array(avg_imp)
    
    # Find first positive
    positive_mask = avg_imp > 0
    if not positive_mask.any():
        return float('inf')  # Never learned
    if positive_mask.all():
        return 0.0  # Always learned
    
    first_positive_idx = np.argmax(positive_mask)
    if first_positive_idx == 0:
        return unique_eps[0]
    
    # Linear interpolation
    eps1 = unique_eps[first_positive_idx - 1]
    eps2 = unique_eps[first_positive_idx]
    imp1 = avg_imp[first_positive_idx - 1]
    imp2 = avg_imp[first_positive_idx]
    
    # eps_crit where imp = 0
    eps_crit = eps1 - imp1 * (eps2 - eps1) / (imp2 - imp1 + 1e-10)
    return float(eps_crit)


def analyze_invariant_patterns(results_file: Path = None) -> Dict[str, Any]:
    """
    Search for invariant patterns across parameter space.
    
    Looks for:
    - Universal scaling collapses
    - Parameter combinations that predict success/failure
    - Dimensionless groups that control behavior
    """
    if results_file is None:
        results_file = Path("results/scaling_laws/cross_parameter_sweep.json")
    
    with open(results_file) as f:
        data = json.load(f)
    
    results = data['results']
    valid = [r for r in results if not r['failed']]
    
    eps = np.array([r['epsilon'] for r in valid])
    n = np.array([r['n_particles'] for r in valid])
    t = np.array([r['horizon'] for r in valid])
    dt = np.array([r['dt'] for r in valid])
    imp = np.array([r['improvement'] for r in valid])
    snr = np.array([r['gradient_snr_mean'] for r in valid])
    
    patterns = {}
    
    # 1. Test dimensionless group: eps * T / N
    dim_group_1 = eps * t / n
    patterns['eps_T_over_N'] = {
        'correlation_with_improvement': float(np.corrcoef(dim_group_1, imp)[0, 1]),
        'correlation_with_snr': float(np.corrcoef(dim_group_1, snr)[0, 1]),
    }
    
    # 2. Test dimensionless group: eps * dt^2 / N
    dim_group_2 = eps * dt**2 / n
    patterns['eps_dt2_over_N'] = {
        'correlation_with_improvement': float(np.corrcoef(dim_group_2, imp)[0, 1]),
        'correlation_with_snr': float(np.corrcoef(dim_group_2, snr)[0, 1]),
    }
    
    # 3. Test SNR threshold
    snr_threshold_candidates = np.linspace(0.1, 2.0, 20)
    best_acc = 0
    best_thresh = 0
    
    for thresh in snr_threshold_candidates:
        pred_success = snr > thresh
        actual_success = imp > 0
        accuracy = np.mean(pred_success == actual_success)
        if accuracy > best_acc:
            best_acc = accuracy
            best_thresh = thresh
    
    patterns['snr_threshold'] = {
        'optimal_threshold': float(best_thresh),
        'prediction_accuracy': float(best_acc),
    }
    
    # 4. Find parameter regime boundaries
    # Use decision tree-like analysis
    patterns['success_regime'] = {
        'min_epsilon_for_success': float(eps[imp > 0].min()) if (imp > 0).any() else None,
        'max_n_for_success': float(n[imp > 0].max()) if (imp > 0).any() else None,
        'max_t_for_success': float(t[imp > 0].max()) if (imp > 0).any() else None,
    }
    
    print("=" * 70)
    print("INVARIANT PATTERN ANALYSIS")
    print("=" * 70)
    print(f"\nDimensionless group eps*T/N correlation with improvement: "
          f"{patterns['eps_T_over_N']['correlation_with_improvement']:.3f}")
    print(f"Dimensionless group eps*dt^2/N correlation with improvement: "
          f"{patterns['eps_dt2_over_N']['correlation_with_improvement']:.3f}")
    print(f"\nSNR threshold for success prediction:")
    print(f"  Optimal threshold: {patterns['snr_threshold']['optimal_threshold']:.3f}")
    print(f"  Prediction accuracy: {patterns['snr_threshold']['prediction_accuracy']:.1%}")
    
    return patterns


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Parameter Scaling Analysis")
    parser.add_argument('--quick', action='store_true', 
                        help='Run quick sweep with fewer configurations')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze existing results')
    parser.add_argument('--output', type=str, default='results/scaling_laws')
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    
    if args.analyze_only:
        analyze_invariant_patterns(output_dir / "cross_parameter_sweep.json")
    elif args.quick:
        cross_parameter_sweep(
            n_values=[5, 10],
            t_values=[20, 50],
            dt_values=[30.0, 60.0],
            epsilon_values=[1.0, 10.0, 50.0],
            n_seeds=1,
            output_dir=output_dir,
        )
    else:
        cross_parameter_sweep(output_dir=output_dir)
