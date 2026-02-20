"""
Theoretical Scaling Law Analysis.

Hypothesis: Learnability ∝ ε × T

Tests whether improvement curves collapse when plotted against
the product of coupling strength and integration horizon.

This establishes a "signal-to-learn" quantity that predicts
when structured correction learning will succeed.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

from h_gnn.models import TwoBodyHamiltonian
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet
from h_gnn.data import SyntheticCatalog
from h_gnn.experiments.epsilon_sweep import train_model_for_epsilon


@dataclass
class ScalingLawResult:
    """Result for a single (ε, T) combination."""
    epsilon: float
    horizon: int
    epsilon_times_T: float
    
    mse_hgnn: float
    mse_twobody: float
    improvement: float
    
    mean_phi: float


def run_scaling_law_analysis(
    epsilons: List[float] = [5, 10, 25, 50, 100],
    horizons: List[int] = [10, 25, 50, 100],
    n_seeds: int = 3,
    save_dir: str = "results/scaling_law",
) -> List[ScalingLawResult]:
    """
    Run ε×T scaling law analysis.
    
    If improvement depends on ε×T, then:
    - Low ε, long T should match high ε, short T
    - Curves should collapse when plotted vs ε×T
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    config = {
        "mu": 398600.4418,
        "dt": 60.0,
        "n_objects": 50,
        "n_epochs": 150,
        "lr": 1e-3,
        "interaction_radius": 5000.0,
    }
    
    mu = config["mu"]
    dt = config["dt"]
    
    print("=" * 70)
    print("ε × T SCALING LAW ANALYSIS")
    print("=" * 70)
    print(f"ε ∈ {epsilons}")
    print(f"T ∈ {horizons}")
    print(f"This tests {len(epsilons) * len(horizons)} (ε, T) combinations")
    print()
    
    all_results = []
    
    # For each epsilon, train a model
    models = {}
    
    for eps in epsilons:
        print(f"\nTraining model for ε={eps}...")
        seed_models = []
        for seed in range(n_seeds):
            model, _, mean_phi = train_model_for_epsilon(eps, config, seed)
            seed_models.append((model, mean_phi))
        models[eps] = seed_models
    
    # Evaluate all (ε, T) combinations
    print("\n" + "=" * 70)
    print("Evaluating all (ε, T) combinations...")
    print("=" * 70)
    
    for eps in epsilons:
        gt_H = MultiBodyHamiltonian(mu=mu, epsilon=eps)
        tb_H = TwoBodyHamiltonian(mu=mu)
        integrator = StormerVerlet(dt=dt)
        
        for horizon in horizons:
            improvements = []
            
            for seed, (model, mean_phi) in enumerate(models[eps]):
                # Test data
                catalog = SyntheticCatalog(mu=mu, seed=seed + 100)
                q_test, p_test = catalog.generate_training_set(100)
                
                with torch.no_grad():
                    q_gt, _ = integrator.integrate(
                        q_test, p_test, gt_H, horizon, store_trajectory=False
                    )
                    q_hgnn, _ = model(q_test.clone(), p_test.clone(), n_steps=horizon)
                    q_tb, _ = integrator.integrate(
                        q_test, p_test, tb_H, horizon, store_trajectory=False
                    )
                
                mse_hgnn = torch.mean((q_hgnn - q_gt) ** 2).item()
                mse_tb = torch.mean((q_tb - q_gt) ** 2).item()
                
                improvement = (1 - mse_hgnn / mse_tb) * 100 if mse_tb > 0 else 0
                improvements.append(improvement)
            
            mean_improvement = np.mean(improvements)
            mean_phi_avg = np.mean([m[1] for m in models[eps]])
            
            result = ScalingLawResult(
                epsilon=eps,
                horizon=horizon,
                epsilon_times_T=eps * horizon,
                mse_hgnn=mse_hgnn,
                mse_twobody=mse_tb,
                improvement=mean_improvement,
                mean_phi=mean_phi_avg,
            )
            
            all_results.append(result)
            
            print(f"ε={eps:3.0f}, T={horizon:3d}, ε×T={eps*horizon:5.0f}: "
                  f"Improvement={mean_improvement:+6.1f}%")
    
    # Analyze scaling
    print("\n" + "=" * 70)
    print("SCALING LAW ANALYSIS")
    print("=" * 70)
    
    # Group by ε×T ranges
    sorted_results = sorted(all_results, key=lambda r: r.epsilon_times_T)
    
    print("\nImprovement vs ε×T:")
    print(f"{'ε×T':>8} {'ε':>6} {'T':>6} {'Improvement':>12}")
    print("-" * 40)
    
    for r in sorted_results:
        print(f"{r.epsilon_times_T:>8.0f} {r.epsilon:>6.0f} {r.horizon:>6} "
              f"{r.improvement:>+11.1f}%")
    
    # Check if similar ε×T values give similar improvements
    print("\n" + "=" * 70)
    print("UNIVERSAL SCALING CHECK")
    print("=" * 70)
    
    # Group by ε×T bins
    bins = [(0, 250), (250, 750), (750, 1500), (1500, 3000), (3000, 10000)]
    
    print(f"\n{'ε×T Range':>15} {'Mean Imp':>12} {'Std':>10} {'N':>5}")
    print("-" * 45)
    
    for low, high in bins:
        in_range = [r for r in all_results if low <= r.epsilon_times_T < high]
        if in_range:
            imps = [r.improvement for r in in_range]
            print(f"{low:>6}-{high:<6} {np.mean(imps):>+11.1f}% "
                  f"{np.std(imps):>9.1f}% {len(imps):>5}")
    
    # Compute correlation
    eT_values = [r.epsilon_times_T for r in all_results]
    improvements = [r.improvement for r in all_results]
    
    # Log-transform for analysis (since improvement may saturate)
    log_eT = np.log(eT_values)
    
    # Pearson correlation
    corr = np.corrcoef(eT_values, improvements)[0, 1]
    log_corr = np.corrcoef(log_eT, improvements)[0, 1]
    
    print(f"\nCorrelation(ε×T, Improvement): {corr:.3f}")
    print(f"Correlation(log(ε×T), Improvement): {log_corr:.3f}")
    
    # Check if curve is monotonic in ε×T (up to saturation)
    # Split into "growth" region (low ε×T) and "saturation" region (high ε×T)
    low_et = [r for r in all_results if r.epsilon_times_T < 1000]
    high_et = [r for r in all_results if r.epsilon_times_T >= 1000]
    
    if low_et:
        low_corr = np.corrcoef([r.epsilon_times_T for r in low_et],
                               [r.improvement for r in low_et])[0, 1]
        print(f"Correlation in growth region (ε×T < 1000): {low_corr:.3f}")
    
    if high_et:
        high_corr = np.corrcoef([r.epsilon_times_T for r in high_et],
                                [r.improvement for r in high_et])[0, 1]
        print(f"Correlation in saturation region (ε×T ≥ 1000): {high_corr:.3f}")
    
    # Save results
    results_file = Path(save_dir) / "scaling_law_results.json"
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    results = run_scaling_law_analysis(
        epsilons=[5, 10, 25, 50, 100],
        horizons=[10, 25, 50, 100],
        n_seeds=3,
    )
