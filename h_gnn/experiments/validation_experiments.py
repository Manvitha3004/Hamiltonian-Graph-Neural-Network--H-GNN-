"""
Validation experiments for backward-error aligned H-GNN architecture.

Criteria for success:
- T=50 > 15% improvement at N=10
- T=50 > 5% improvement at N=25  
- α converges to similar magnitude across seeds
"""

import torch
import numpy as np
from typing import Dict, List
import time

from h_gnn.experiments.backward_error_training import (
    train_backward_error_aligned,
    evaluate_backward_error_model,
)


def run_validation_experiments():
    """Run validation experiments for architecture verification."""
    
    base_config = {
        'mu': 398600.4418,
        'dt_primary': 60.0,
        'dt_secondary': 30.0,
        'n_epochs': 200,
        'lr': 5e-5,
        'epsilon': 100.0,
        'interaction_radius': 20000.0,
        'batch_size': 16,
        'horizons': [5, 10, 20, 50],
        'weights': [0.2, 0.3, 0.3, 0.2],
        'lambda_drift': 0.1,
        'use_dual_dt': False,
        'n_eval': 20,
    }
    
    results_all = {}
    
    # ============================================================
    # Experiment 1: N=10, 3 seeds
    # ============================================================
    print("=" * 60)
    print("EXPERIMENT 1: N=10, 3 seeds, T=50")
    print("=" * 60)
    
    n10_results = []
    n10_alphas = []
    seeds_n10 = [42, 123, 456]
    
    for seed in seeds_n10:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        config = base_config.copy()
        config['n_particles'] = 10
        
        t_start = time.time()
        model = train_backward_error_aligned(config)
        train_time = time.time() - t_start
        
        # Get final alpha
        final_alpha = model.hamiltonian.alpha.item()
        n10_alphas.append(final_alpha)
        
        # Evaluate
        results = evaluate_backward_error_model(model, config)
        
        t50_improvement = results.get(50, {}).get('improvement', 0)
        n10_results.append(t50_improvement)
        
        print(f"  T=50 improvement: {t50_improvement:.2f}%")
        print(f"  Final alpha: {final_alpha:.6e}")
        print(f"  Training time: {train_time:.1f}s")
    
    # Summary N=10
    print("\n" + "=" * 60)
    print("N=10 SUMMARY")
    print("=" * 60)
    mean_imp_n10 = np.mean(n10_results)
    std_imp_n10 = np.std(n10_results)
    mean_alpha_n10 = np.mean(n10_alphas)
    std_alpha_n10 = np.std(n10_alphas)
    
    print(f"T=50 improvements: {n10_results}")
    print(f"Mean T=50: {mean_imp_n10:.2f}% ± {std_imp_n10:.2f}%")
    print(f"Final alphas: {[f'{a:.2e}' for a in n10_alphas]}")
    print(f"Mean alpha: {mean_alpha_n10:.2e} ± {std_alpha_n10:.2e}")
    
    pass_n10 = mean_imp_n10 > 15.0
    print(f"\nCRITERION: T=50 > 15% at N=10: {'PASS' if pass_n10 else 'FAIL'} ({mean_imp_n10:.2f}%)")
    
    results_all['N10'] = {
        'improvements': n10_results,
        'alphas': n10_alphas,
        'mean_improvement': mean_imp_n10,
        'std_improvement': std_imp_n10,
        'mean_alpha': mean_alpha_n10,
        'std_alpha': std_alpha_n10,
        'pass': pass_n10,
    }
    
    # ============================================================
    # Experiment 2: N=25, 1 seed
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: N=25, 1 seed, T=50")
    print("=" * 60)
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    config = base_config.copy()
    config['n_particles'] = 25
    config['n_epochs'] = 200  # Same epochs
    
    t_start = time.time()
    model = train_backward_error_aligned(config)
    train_time = time.time() - t_start
    
    final_alpha_n25 = model.hamiltonian.alpha.item()
    results = evaluate_backward_error_model(model, config)
    
    t50_improvement_n25 = results.get(50, {}).get('improvement', 0)
    
    print(f"\n  T=50 improvement: {t50_improvement_n25:.2f}%")
    print(f"  Final alpha: {final_alpha_n25:.6e}")
    print(f"  Training time: {train_time:.1f}s")
    
    pass_n25 = t50_improvement_n25 > 5.0
    print(f"\nCRITERION: T=50 > 5% at N=25: {'PASS' if pass_n25 else 'FAIL'} ({t50_improvement_n25:.2f}%)")
    
    results_all['N25'] = {
        'improvement': t50_improvement_n25,
        'alpha': final_alpha_n25,
        'pass': pass_n25,
    }
    
    # ============================================================
    # Alpha convergence check
    # ============================================================
    print("\n" + "=" * 60)
    print("ALPHA CONVERGENCE CHECK")
    print("=" * 60)
    
    all_alphas = n10_alphas + [final_alpha_n25]
    alpha_mean = np.mean(all_alphas)
    alpha_std = np.std(all_alphas)
    alpha_cv = alpha_std / abs(alpha_mean) if alpha_mean != 0 else float('inf')
    
    print(f"All final alphas: {[f'{a:.2e}' for a in all_alphas]}")
    print(f"Mean: {alpha_mean:.2e}")
    print(f"Std: {alpha_std:.2e}")
    print(f"CV (std/mean): {alpha_cv:.2f}")
    
    # Consider converged if CV < 1.0 (within same order of magnitude)
    alpha_converged = alpha_cv < 1.0
    print(f"\nCRITERION: alpha converges to similar magnitude: {'PASS' if alpha_converged else 'FAIL'} (CV={alpha_cv:.2f})")
    
    results_all['alpha_convergence'] = {
        'all_alphas': all_alphas,
        'mean': alpha_mean,
        'std': alpha_std,
        'cv': alpha_cv,
        'pass': alpha_converged,
    }
    
    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    
    all_pass = pass_n10 and pass_n25 and alpha_converged
    
    print(f"N=10 T=50 > 15%:  {'PASS' if pass_n10 else 'FAIL'} ({mean_imp_n10:.2f}%)")
    print(f"N=25 T=50 > 5%:   {'PASS' if pass_n25 else 'FAIL'} ({t50_improvement_n25:.2f}%)")
    print(f"alpha convergence: {'PASS' if alpha_converged else 'FAIL'} (CV={alpha_cv:.2f})")
    print()
    
    if all_pass:
        print("ALL CRITERIA PASSED - Architecture is validated!")
    else:
        print("Some criteria not met - Further tuning may be needed")
    
    return results_all


if __name__ == "__main__":
    results = run_validation_experiments()
