"""
Run Critical Experiments for Backward-Error Aligned H-GNN.

Three critical experiments:
1. N-scaling test: N=10, 25, 3 seeds each
2. dt-scaling test: train at dt=60, evaluate at dt=60 and dt=30
3. T=100 multi-seed validation: 3 seeds, check variance < 15%
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from h_gnn.experiments.multiscale_training import train_multiscale, evaluate_multiscale


def run_n_scaling_test(base_config: dict, n_values=[10, 25], n_seeds=3):
    """
    N-scaling test: With 1/N normalization, improvement should be stable across N.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: N-SCALING TEST")
    print("=" * 70)
    
    results = {}
    
    for n in n_values:
        print(f"\n--- N = {n} ---")
        improvements = []
        
        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}")
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            cfg = base_config.copy()
            cfg["n_particles"] = n
            cfg["n_epochs"] = 150
            cfg["batch_size"] = 8
            
            model = train_multiscale(cfg)
            eval_results = evaluate_multiscale(model, cfg)
            
            # Get T=50 improvement
            improvement = eval_results.get(50, {}).get("improvement", 0.0)
            improvements.append(improvement)
            print(f"    T=50 improvement: {improvement:.1f}%")
        
        results[n] = {
            "mean": np.mean(improvements),
            "std": np.std(improvements),
            "values": improvements,
        }
        print(f"  N={n} result: {np.mean(improvements):.1f}% ± {np.std(improvements):.1f}%")
    
    print("\n" + "-" * 70)
    print("N-SCALING SUMMARY")
    print("-" * 70)
    for n, r in results.items():
        print(f"  N={n}: {r['mean']:.1f}% ± {r['std']:.1f}%")
    
    return results


def run_dt_scaling_test(base_config: dict):
    """
    dt-scaling test: MSE(dt=60) / MSE(dt=30) should be ~4 for dt² scaling.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: dt-SCALING TEST")
    print("=" * 70)
    
    from h_gnn.models.hgnn import HGNN
    from h_gnn.models.multibody import MultiBodyHamiltonian
    from h_gnn.models.hamiltonian import TwoBodyHamiltonian
    from h_gnn.integrators.symplectic import StormerVerlet
    
    cfg = base_config.copy()
    cfg["n_epochs"] = 200
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train at dt=60
    print("\nTraining at dt=60...")
    model = train_multiscale(cfg)
    
    # Evaluate at both dt values
    mu = cfg.get("mu", 398600.4418)
    epsilon = cfg.get("epsilon", 100.0)
    n_particles = cfg.get("n_particles", 5)
    n_eval = 10
    
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    def generate_ic(seed):
        rng = np.random.default_rng(seed)
        q_list, p_list = [], []
        for _ in range(n_particles):
            r = rng.uniform(7000, 42000)
            theta = rng.uniform(0, 2 * np.pi)
            q = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
            v_circ = np.sqrt(mu / r)
            p = v_circ * np.array([-np.sin(theta), np.cos(theta), 0.0])
            q_list.append(q)
            p_list.append(p)
        return torch.tensor(np.stack(q_list), dtype=torch.float64), \
               torch.tensor(np.stack(p_list), dtype=torch.float64)
    
    results = {}
    
    for dt in [60.0, 30.0]:
        print(f"\nEvaluating at dt={dt}...")
        model.integrator.dt.fill_(dt)
        integrator = StormerVerlet(dt=dt, create_graph=False)
        
        # Scale horizons to match physical time
        if dt == 30.0:
            horizons = [50, 100, 200]  # Same physical time as [25, 50, 100] at dt=60
        else:
            horizons = [25, 50, 100]
        
        mse_by_time = {}
        
        for T in horizons:
            physical_time = T * dt
            mse_list = []
            
            for seed in range(n_eval):
                q0, p0 = generate_ic(seed + 1000)
                
                with torch.no_grad():
                    # Ground truth
                    q_gt, p_gt = q0.clone(), p0.clone()
                    for _ in range(T):
                        q_gt, p_gt = integrator.step(q_gt, p_gt, gt_hamiltonian, edge_index)
                    
                    # Model
                    q_pred, p_pred = model.forward(
                        q0.float(), p0.float(), n_steps=T, return_trajectory=False
                    )
                    
                    mse = torch.mean((q_pred - q_gt.float())**2).item()
                    mse_list.append(mse)
            
            mse_by_time[physical_time] = np.mean(mse_list)
        
        results[dt] = mse_by_time
    
    # Compare at same physical time
    print("\n" + "-" * 70)
    print("dt-SCALING COMPARISON (same physical time)")
    print("-" * 70)
    print(f"{'Physical Time':>15} {'MSE(dt=60)':>12} {'MSE(dt=30)':>12} {'Ratio':>10}")
    
    common_times = sorted(set(results[60.0].keys()) & set(results[30.0].keys()))
    ratios = []
    
    for t in common_times:
        mse_60 = results[60.0][t]
        mse_30 = results[30.0][t]
        ratio = mse_60 / mse_30 if mse_30 > 0 else float('inf')
        ratios.append(ratio)
        print(f"{t:>15.0f}s {mse_60:>12.4e} {mse_30:>12.4e} {ratio:>10.2f}")
    
    mean_ratio = np.mean(ratios) if ratios else 0
    print(f"\nMean ratio: {mean_ratio:.2f} (expected ~4 for dt² scaling)")
    
    return {"ratios": ratios, "mean_ratio": mean_ratio, "results": results}


def run_t100_multiseed_test(base_config: dict, n_seeds=3):
    """
    T=100 multi-seed validation: check if improvement is positive with variance < 15%.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: T=100 MULTI-SEED VALIDATION")
    print("=" * 70)
    
    improvements = []
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        cfg = base_config.copy()
        cfg["n_epochs"] = 200
        
        model = train_multiscale(cfg)
        eval_results = evaluate_multiscale(model, cfg)
        
        improvement = eval_results.get(100, {}).get("improvement", 0.0)
        improvements.append(improvement)
        print(f"  T=100 improvement: {improvement:.1f}%")
    
    mean_imp = np.mean(improvements)
    std_imp = np.std(improvements)
    all_positive = all(x > 0 for x in improvements)
    
    print("\n" + "-" * 70)
    print("T=100 MULTI-SEED SUMMARY")
    print("-" * 70)
    print(f"  Mean: {mean_imp:.1f}%")
    print(f"  Std: {std_imp:.1f}%")
    print(f"  All positive: {all_positive}")
    print(f"  Variance acceptable (<15%): {std_imp < 15}")
    
    return {
        "mean": mean_imp,
        "std": std_imp,
        "values": improvements,
        "all_positive": all_positive,
        "variance_acceptable": std_imp < 15,
    }


def main():
    """Run all critical experiments."""
    print("=" * 70)
    print("CRITICAL EXPERIMENTS FOR H-GNN")
    print("=" * 70)
    
    # Base configuration
    base_config = {
        "mu": 398600.4418,
        "dt": 60.0,
        "n_particles": 5,
        "n_epochs": 200,
        "lr": 1e-4,
        "epsilon": 100.0,
        "interaction_radius": 20000.0,
        "batch_size": 10,
        "horizons": [5, 10, 20, 50],
        "weights": [0.2, 0.3, 0.3, 0.2],
        "lambda_energy": 0.1,
        "phi_scale": 0.005,
        "n_eval": 10,
    }
    
    all_results = {}
    
    # Experiment 1: N-scaling
    print("\n" + "=" * 70)
    print("Starting N-scaling test...")
    n_scaling_results = run_n_scaling_test(base_config, n_values=[10, 25], n_seeds=3)
    all_results["n_scaling"] = n_scaling_results
    
    # Experiment 2: dt-scaling
    print("\n" + "=" * 70)
    print("Starting dt-scaling test...")
    dt_scaling_results = run_dt_scaling_test(base_config)
    all_results["dt_scaling"] = dt_scaling_results
    
    # Experiment 3: T=100 multi-seed
    print("\n" + "=" * 70)
    print("Starting T=100 multi-seed test...")
    t100_results = run_t100_multiseed_test(base_config, n_seeds=3)
    all_results["t100_multiseed"] = t100_results
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\n1. N-SCALING:")
    for n, r in n_scaling_results.items():
        print(f"   N={n}: {r['mean']:.1f}% ± {r['std']:.1f}%")
    
    print(f"\n2. dt-SCALING:")
    print(f"   Mean ratio: {dt_scaling_results['mean_ratio']:.2f} (expected ~4)")
    
    print(f"\n3. T=100 MULTI-SEED:")
    print(f"   Mean: {t100_results['mean']:.1f}% ± {t100_results['std']:.1f}%")
    print(f"   All positive: {t100_results['all_positive']}")
    print(f"   Variance acceptable: {t100_results['variance_acceptable']}")
    
    return all_results


if __name__ == "__main__":
    main()
