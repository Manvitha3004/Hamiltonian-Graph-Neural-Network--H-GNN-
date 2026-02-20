"""
Final clean validation experiments for H-GNN backward-error aligned architecture.

Runs:
- N=10 with 3 seeds
- N=25 with 1 seed

Reports mean +/- std for all metrics.
"""

import torch
import numpy as np
import json
import time

from h_gnn.experiments.backward_error_training import (
    train_backward_error_aligned,
    evaluate_backward_error_model,
)


def run_final_validation():
    """Run clean validation experiments and report results."""
    
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

    results = {}
    
    # ================================================================
    # N=10, 3 seeds
    # ================================================================
    print("=" * 60)
    print("EXPERIMENT: N=10, 3 seeds")
    print("=" * 60)
    
    n10_improvements = []
    n10_alphas = []
    seeds = [42, 123, 456]
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        config = base_config.copy()
        config['n_particles'] = 10
        
        t0 = time.time()
        model = train_backward_error_aligned(config)
        train_time = time.time() - t0
        
        alpha_final = model.hamiltonian.alpha.item()
        n10_alphas.append(alpha_final)
        
        eval_results = evaluate_backward_error_model(model, config)
        imp50 = eval_results.get(50, {}).get('improvement', 0)
        n10_improvements.append(imp50)
        
        print(f"  T=50 improvement: {imp50:.2f}%")
        print(f"  Final alpha: {alpha_final:.4e}")
        print(f"  Train time: {train_time:.1f}s")
    
    # N=10 Summary
    print("\n" + "=" * 60)
    print("N=10 SUMMARY")
    print("=" * 60)
    mean_n10 = np.mean(n10_improvements)
    std_n10 = np.std(n10_improvements)
    mean_alpha_n10 = np.mean(n10_alphas)
    std_alpha_n10 = np.std(n10_alphas)
    
    print(f"T=50 improvements: {[f'{x:.2f}%' for x in n10_improvements]}")
    print(f"Mean: {mean_n10:.2f}% +/- {std_n10:.2f}%")
    print(f"Final alphas: {[f'{x:.2e}' for x in n10_alphas]}")
    print(f"Alpha mean: {mean_alpha_n10:.2e} +/- {std_alpha_n10:.2e}")
    
    results['N10'] = {
        'improvements': n10_improvements,
        'alphas': n10_alphas,
        'mean_improvement': float(mean_n10),
        'std_improvement': float(std_n10),
        'mean_alpha': float(mean_alpha_n10),
        'std_alpha': float(std_alpha_n10),
    }
    
    # ================================================================
    # N=25, 1 seed
    # ================================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT: N=25, seed=42")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    config = base_config.copy()
    config['n_particles'] = 25
    
    t0 = time.time()
    model = train_backward_error_aligned(config)
    train_time = time.time() - t0
    
    alpha_n25 = model.hamiltonian.alpha.item()
    eval_results = evaluate_backward_error_model(model, config)
    imp50_n25 = eval_results.get(50, {}).get('improvement', 0)
    
    print(f"T=50 improvement: {imp50_n25:.2f}%")
    print(f"Final alpha: {alpha_n25:.4e}")
    print(f"Train time: {train_time:.1f}s")
    
    results['N25'] = {
        'improvement': float(imp50_n25),
        'alpha': float(alpha_n25),
    }
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    all_alphas = n10_alphas + [alpha_n25]
    alpha_cv = np.std(all_alphas) / abs(np.mean(all_alphas))
    
    print(f"\nN=10 (3 seeds):")
    print(f"  T=50: {mean_n10:.2f}% +/- {std_n10:.2f}%")
    print(f"  Alpha: {mean_alpha_n10:.2e} +/- {std_alpha_n10:.2e}")
    
    print(f"\nN=25 (1 seed):")
    print(f"  T=50: {imp50_n25:.2f}%")
    print(f"  Alpha: {alpha_n25:.2e}")
    
    print(f"\nAlpha convergence:")
    print(f"  All alphas: {[f'{a:.2e}' for a in all_alphas]}")
    print(f"  CV (std/mean): {alpha_cv:.3f}")
    
    # Criteria check
    print("\n" + "-" * 60)
    print("CRITERIA CHECK")
    print("-" * 60)
    
    pass_n10 = mean_n10 > 15.0
    pass_n25 = imp50_n25 > 5.0
    alpha_ok = alpha_cv < 1.0
    
    print(f"N=10 T=50 > 15%: {'PASS' if pass_n10 else 'FAIL'} ({mean_n10:.2f}%)")
    print(f"N=25 T=50 > 5%:  {'PASS' if pass_n25 else 'FAIL'} ({imp50_n25:.2f}%)")
    print(f"Alpha CV < 1.0:  {'PASS' if alpha_ok else 'FAIL'} (CV={alpha_cv:.3f})")
    
    all_pass = pass_n10 and pass_n25 and alpha_ok
    print(f"\nOVERALL: {'ALL CRITERIA PASSED' if all_pass else 'SOME CRITERIA FAILED'}")
    
    results['summary'] = {
        'pass_n10': pass_n10,
        'pass_n25': pass_n25,
        'alpha_converged': alpha_ok,
        'all_pass': all_pass,
    }
    
    # Save results
    with open('final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to final_results.json")
    
    return results


if __name__ == "__main__":
    results = run_final_validation()
