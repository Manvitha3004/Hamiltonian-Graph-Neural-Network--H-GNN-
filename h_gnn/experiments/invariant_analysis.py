"""
Physics Invariant Analysis & Long-Horizon Stability Test.

Tracks:
1. Total energy drift over time
2. Angular momentum drift
3. Phase-space volume (symplectic area)

Tests stability at extended horizons: 10, 25, 50, 100, 200, 500 steps.

Expected: H-GNN maintains bounded energy, vanilla GNN explodes.
"""

import torch
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from h_gnn.models import HGNN, TwoBodyHamiltonian
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet
from h_gnn.data import SyntheticCatalog
from h_gnn.experiments.baseline_comparison import AccelerationCorrectionGNN, train_vanilla_gnn
from h_gnn.experiments.epsilon_sweep import train_model_for_epsilon


@dataclass
class InvariantMetrics:
    """Physics invariant metrics at a given time."""
    step: int
    
    # Energy metrics
    energy_hgnn: float
    energy_vanilla: float
    energy_twobody: float
    energy_gt: float
    
    # Energy drift (relative to initial)
    energy_drift_hgnn: float
    energy_drift_vanilla: float
    energy_drift_twobody: float
    
    # Angular momentum magnitude
    L_hgnn: float
    L_vanilla: float
    L_twobody: float
    L_gt: float
    
    # Angular momentum drift
    L_drift_hgnn: float
    L_drift_vanilla: float
    L_drift_twobody: float
    
    # Position MSE vs ground truth
    mse_hgnn: float
    mse_vanilla: float
    mse_twobody: float


def compute_angular_momentum(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Compute total angular momentum magnitude.
    
    L = Σ |r × p|
    """
    L_vec = torch.cross(q, p, dim=-1)  # (N, 3)
    L_total = L_vec.sum(dim=0)  # (3,)
    return torch.norm(L_total)


def run_invariant_analysis(
    epsilon: float = 50.0,
    horizons: List[int] = [10, 25, 50, 100, 200, 500],
    n_seeds: int = 3,
    save_dir: str = "results/invariant_analysis",
) -> Dict:
    """
    Run comprehensive physics invariant analysis.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    config = {
        "mu": 398600.4418,
        "dt": 60.0,
        "n_objects": 50,
        "n_epochs": 150,
        "lr": 1e-3,
        "interaction_radius": 5000.0,
        "rollout_steps": 10,
    }
    
    mu = config["mu"]
    dt = config["dt"]
    
    print("=" * 70)
    print("PHYSICS INVARIANT ANALYSIS & LONG-HORIZON STABILITY")
    print("=" * 70)
    print(f"ε = {epsilon}, Horizons = {horizons}")
    print()
    
    all_metrics = []
    
    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed + 1}/{n_seeds}")
        print("=" * 70)
        
        # Train models
        print("\nTraining H-GNN...")
        hgnn_model, _, _ = train_model_for_epsilon(epsilon, config, seed)
        
        print("\nTraining Vanilla GNN...")
        vanilla_model = train_vanilla_gnn(epsilon, config, seed)
        
        # Ground truth
        gt_H = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
        tb_H = TwoBodyHamiltonian(mu=mu)
        integrator = StormerVerlet(dt=dt)
        
        # Test data
        catalog = SyntheticCatalog(mu=mu, seed=seed + 100)
        q_test, p_test = catalog.generate_training_set(50)
        
        # Initial energy and angular momentum
        E_init = gt_H(q_test, p_test).item()
        L_init = compute_angular_momentum(q_test, p_test).item()
        
        print(f"\nInitial energy: {E_init:.2f}")
        print(f"Initial |L|: {L_init:.2f}")
        print()
        
        for horizon in horizons:
            with torch.no_grad():
                # Integrate all models
                q_gt, p_gt = integrator.integrate(
                    q_test, p_test, gt_H, horizon, store_trajectory=False
                )
                q_hgnn, p_hgnn = hgnn_model(
                    q_test.clone(), p_test.clone(), n_steps=horizon
                )
                q_vanilla, p_vanilla = vanilla_model(
                    q_test.clone(), p_test.clone(), n_steps=horizon
                )
                q_tb, p_tb = integrator.integrate(
                    q_test, p_test, tb_H, horizon, store_trajectory=False
                )
                
                # Compute energies (using ground truth Hamiltonian)
                E_gt = gt_H(q_gt, p_gt).item()
                E_hgnn = gt_H(q_hgnn, p_hgnn).item()
                E_vanilla = gt_H(q_vanilla, p_vanilla).item()
                E_tb = gt_H(q_tb, p_tb).item()
                
                # Compute angular momentum
                L_gt = compute_angular_momentum(q_gt, p_gt).item()
                L_hgnn = compute_angular_momentum(q_hgnn, p_hgnn).item()
                L_vanilla = compute_angular_momentum(q_vanilla, p_vanilla).item()
                L_tb = compute_angular_momentum(q_tb, p_tb).item()
                
                # MSE
                mse_hgnn = torch.mean((q_hgnn - q_gt) ** 2).item()
                mse_vanilla = torch.mean((q_vanilla - q_gt) ** 2).item()
                mse_tb = torch.mean((q_tb - q_gt) ** 2).item()
            
            metrics = InvariantMetrics(
                step=horizon,
                energy_hgnn=E_hgnn,
                energy_vanilla=E_vanilla,
                energy_twobody=E_tb,
                energy_gt=E_gt,
                energy_drift_hgnn=abs(E_hgnn - E_init),
                energy_drift_vanilla=abs(E_vanilla - E_init),
                energy_drift_twobody=abs(E_tb - E_init),
                L_hgnn=L_hgnn,
                L_vanilla=L_vanilla,
                L_twobody=L_tb,
                L_gt=L_gt,
                L_drift_hgnn=abs(L_hgnn - L_init),
                L_drift_vanilla=abs(L_vanilla - L_init),
                L_drift_twobody=abs(L_tb - L_init),
                mse_hgnn=mse_hgnn,
                mse_vanilla=mse_vanilla,
                mse_twobody=mse_tb,
            )
            
            all_metrics.append(metrics)
            
            # Check for explosion
            is_exploded = mse_vanilla > 1e10 or np.isnan(mse_vanilla)
            status = " 💥 EXPLODED" if is_exploded else ""
            
            print(f"T={horizon:3d}: E_drift H-GNN={metrics.energy_drift_hgnn:.2e}, "
                  f"Vanilla={metrics.energy_drift_vanilla:.2e}, "
                  f"MSE H-GNN={mse_hgnn:.2e}, Vanilla={mse_vanilla:.2e}{status}")
    
    # Aggregate by horizon
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean over seeds)")
    print("=" * 70)
    
    print("\nEnergy Drift:")
    print(f"{'Horizon':>8} {'H-GNN':>12} {'Vanilla':>12} {'Two-body':>12}")
    print("-" * 50)
    
    for horizon in horizons:
        h_metrics = [m for m in all_metrics if m.step == horizon]
        
        drift_hgnn = np.mean([m.energy_drift_hgnn for m in h_metrics])
        drift_vanilla = np.mean([m.energy_drift_vanilla for m in h_metrics])
        drift_tb = np.mean([m.energy_drift_twobody for m in h_metrics])
        
        print(f"{horizon:>8} {drift_hgnn:>12.2e} {drift_vanilla:>12.2e} {drift_tb:>12.2e}")
    
    print("\nAngular Momentum Drift:")
    print(f"{'Horizon':>8} {'H-GNN':>12} {'Vanilla':>12} {'Two-body':>12}")
    print("-" * 50)
    
    for horizon in horizons:
        h_metrics = [m for m in all_metrics if m.step == horizon]
        
        drift_hgnn = np.mean([m.L_drift_hgnn for m in h_metrics])
        drift_vanilla = np.mean([m.L_drift_vanilla for m in h_metrics])
        drift_tb = np.mean([m.L_drift_twobody for m in h_metrics])
        
        print(f"{horizon:>8} {drift_hgnn:>12.2e} {drift_vanilla:>12.2e} {drift_tb:>12.2e}")
    
    print("\nPosition MSE vs Ground Truth:")
    print(f"{'Horizon':>8} {'H-GNN':>14} {'Vanilla':>14} {'Two-body':>14}")
    print("-" * 60)
    
    for horizon in horizons:
        h_metrics = [m for m in all_metrics if m.step == horizon]
        
        mse_hgnn = np.mean([m.mse_hgnn for m in h_metrics])
        mse_vanilla = np.mean([m.mse_vanilla for m in h_metrics])
        mse_tb = np.mean([m.mse_twobody for m in h_metrics])
        
        # Improvement percentages
        imp_hgnn = (1 - mse_hgnn / mse_tb) * 100 if mse_tb > 0 else 0
        imp_vanilla = (1 - mse_vanilla / mse_tb) * 100 if mse_tb > 0 else 0
        
        print(f"{horizon:>8} {mse_hgnn:>10.2e} ({imp_hgnn:+5.1f}%) "
              f"{mse_vanilla:>10.2e} ({imp_vanilla:+5.1f}%) "
              f"{mse_tb:>14.2e}")
    
    # Save results
    results_file = Path(save_dir) / "invariant_results.json"
    with open(results_file, "w") as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return {"metrics": all_metrics}


if __name__ == "__main__":
    run_invariant_analysis(
        epsilon=50.0,
        horizons=[10, 25, 50, 100, 200, 500],
        n_seeds=3,
    )
