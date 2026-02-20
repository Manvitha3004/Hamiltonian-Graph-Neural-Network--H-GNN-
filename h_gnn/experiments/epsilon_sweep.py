"""
ε-Sweep Scaling Law Experiment.

Quantify the learnability threshold: how strong must the
multi-body coupling be for Φ to learn meaningful corrections?

Expected behavior:
- Below threshold: improvement ≈ 0 (Φ collapses)
- Above threshold: improvement increases smoothly
- Very large ε: potential instability

This establishes a scaling law for structured correction learning.
"""

import torch
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from h_gnn.models import HGNN, TwoBodyHamiltonian
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet
from h_gnn.data import SyntheticCatalog


@dataclass
class EpsilonSweepResult:
    """Results for a single ε value."""
    epsilon: float
    
    # Training metrics
    final_loss: float
    mean_phi_magnitude: float
    
    # Rollout MSE (vs multi-body ground truth)
    mse_hgnn_10: float
    mse_hgnn_25: float
    mse_hgnn_50: float
    mse_twobody_10: float
    mse_twobody_25: float
    mse_twobody_50: float
    
    # Improvement percentages
    improvement_10: float
    improvement_25: float
    improvement_50: float
    
    # Energy drift
    energy_drift_hgnn: float
    energy_drift_twobody: float
    
    # Rare-regime improvement (polar)
    rare_regime_improvement: float


def train_model_for_epsilon(
    epsilon: float,
    config: Dict,
    seed: int = 42,
) -> Tuple[HGNN, float, float]:
    """
    Train H-GNN model for a specific ε value.
    
    Returns:
        (model, final_loss, mean_phi_magnitude)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    mu = config["mu"]
    dt = config["dt"]
    n_objects = config["n_objects"]
    n_epochs = config["n_epochs"]
    lr = config["lr"]
    interaction_radius = config["interaction_radius"]
    
    # Generate training data
    catalog = SyntheticCatalog(mu=mu, seed=seed)
    q_train, p_train = catalog.generate_training_set(n_objects)
    
    # Hamiltonians
    gt_H = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    tb_H = TwoBodyHamiltonian(mu=mu)
    
    # Model
    model = HGNN(
        mu=mu,
        interaction_strength=1.0,
        interaction_radius=interaction_radius,
        hidden_dims=[128, 128, 64],
        integrator_type="stormer_verlet",
        dt=dt,
        create_graph=True,
        softplus_output=False,
        normalize_features=True,
    ).double()
    
    # Build graph
    graph = model.graph_builder(q_train)
    edge_index = graph.edge_index
    
    # Target gradients
    grad_q_gt, _ = gt_H.compute_gradients(q_train, p_train)
    grad_q_tb, _ = tb_H.compute_gradients(q_train, p_train)
    target_grad = grad_q_gt - grad_q_tb
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        
        grad_q_model, _ = model.hamiltonian.compute_gradients(
            q_train, p_train, edge_index, create_graph=True
        )
        grad_q_tb_curr, _ = tb_H.compute_gradients(q_train, p_train)
        model_correction = grad_q_model - grad_q_tb_curr
        
        loss = torch.mean((model_correction - target_grad) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Restore best model
    model.load_state_dict(best_state)
    model.eval()
    
    # Compute mean |Φ|
    with torch.no_grad():
        src, dst = edge_index
        q_i, q_j = q_train[src], q_train[dst]
        p_i, p_j = p_train[src], p_train[dst]
        phi_values = model.interaction(q_i, p_i, q_j, p_j)
        mean_phi = phi_values.abs().mean().item()
    
    return model, best_loss, mean_phi


def evaluate_model(
    model: HGNN,
    epsilon: float,
    config: Dict,
    seed: int = 99,
) -> Dict:
    """Evaluate model on test data."""
    mu = config["mu"]
    dt = config["dt"]
    
    gt_H = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    tb_H = TwoBodyHamiltonian(mu=mu)
    integrator = StormerVerlet(dt=dt)
    
    # Test data (different seed)
    catalog = SyntheticCatalog(mu=mu, seed=seed)
    q_test, p_test = catalog.generate_training_set(100)
    
    results = {}
    
    # Evaluate at multiple horizons
    for n_steps in [10, 25, 50]:
        with torch.no_grad():
            q_gt, p_gt = integrator.integrate(q_test, p_test, gt_H, n_steps, store_trajectory=True)
            q_hgnn, p_hgnn = model(q_test.clone(), p_test.clone(), n_steps=n_steps, return_trajectory=True)
            q_tb, p_tb = integrator.integrate(q_test, p_test, tb_H, n_steps, store_trajectory=True)
        
        mse_hgnn = torch.mean((q_hgnn[-1] - q_gt[-1])**2).item()
        mse_tb = torch.mean((q_tb[-1] - q_gt[-1])**2).item()
        
        results[f"mse_hgnn_{n_steps}"] = mse_hgnn
        results[f"mse_twobody_{n_steps}"] = mse_tb
        results[f"improvement_{n_steps}"] = (1 - mse_hgnn / mse_tb) * 100 if mse_tb > 0 else 0
        
        # Energy drift at 50 steps
        if n_steps == 50:
            E_gt_init = gt_H(q_gt[0], p_gt[0])
            E_gt_final = gt_H(q_gt[-1], p_gt[-1])
            E_hgnn_init = gt_H(q_hgnn[0], p_hgnn[0])
            E_hgnn_final = gt_H(q_hgnn[-1], p_hgnn[-1])
            E_tb_init = gt_H(q_tb[0], p_tb[0])
            E_tb_final = gt_H(q_tb[-1], p_tb[-1])
            
            results["energy_drift_hgnn"] = abs(E_hgnn_final - E_hgnn_init).item()
            results["energy_drift_twobody"] = abs(E_tb_final - E_tb_init).item()
    
    # Rare-regime test (polar orbits)
    q_polar, p_polar = catalog.generate_training_set(
        n_objects=50,
        inclination_range=(np.radians(70), np.radians(90))
    )
    
    with torch.no_grad():
        q_gt_polar, _ = integrator.integrate(q_polar, p_polar, gt_H, 50, store_trajectory=False)
        q_hgnn_polar, _ = model(q_polar.clone(), p_polar.clone(), n_steps=50)
        q_tb_polar, _ = integrator.integrate(q_polar, p_polar, tb_H, 50, store_trajectory=False)
    
    mse_hgnn_polar = torch.mean((q_hgnn_polar - q_gt_polar)**2).item()
    mse_tb_polar = torch.mean((q_tb_polar - q_gt_polar)**2).item()
    results["rare_regime_improvement"] = (1 - mse_hgnn_polar / mse_tb_polar) * 100 if mse_tb_polar > 0 else 0
    
    return results


def run_epsilon_sweep(
    epsilons: List[float] = [1, 5, 10, 25, 50, 100, 200],
    n_seeds: int = 3,
    save_dir: str = "results/epsilon_sweep",
) -> List[EpsilonSweepResult]:
    """
    Run complete ε-sweep experiment.
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
    
    print("=" * 70)
    print("ε-SWEEP SCALING LAW EXPERIMENT")
    print("=" * 70)
    print(f"Testing ε ∈ {epsilons}")
    print(f"Seeds: {n_seeds}, Epochs: {config['n_epochs']}")
    print()
    
    all_results = []
    
    for eps in epsilons:
        print(f"\n{'='*70}")
        print(f"ε = {eps}")
        print("=" * 70)
        
        # Aggregate over seeds
        seed_results = []
        
        for seed in range(n_seeds):
            print(f"  Seed {seed+1}/{n_seeds}...", end=" ", flush=True)
            
            # Train
            model, final_loss, mean_phi = train_model_for_epsilon(
                epsilon=eps, config=config, seed=seed
            )
            
            # Evaluate
            eval_results = evaluate_model(model, eps, config, seed=seed+100)
            
            seed_results.append({
                "final_loss": final_loss,
                "mean_phi_magnitude": mean_phi,
                **eval_results,
            })
            
            print(f"Loss={final_loss:.2e}, |Φ|={mean_phi:.4f}, "
                  f"Imp@50={eval_results['improvement_50']:.1f}%")
        
        # Average over seeds
        avg_result = EpsilonSweepResult(
            epsilon=eps,
            final_loss=np.mean([r["final_loss"] for r in seed_results]),
            mean_phi_magnitude=np.mean([r["mean_phi_magnitude"] for r in seed_results]),
            mse_hgnn_10=np.mean([r["mse_hgnn_10"] for r in seed_results]),
            mse_hgnn_25=np.mean([r["mse_hgnn_25"] for r in seed_results]),
            mse_hgnn_50=np.mean([r["mse_hgnn_50"] for r in seed_results]),
            mse_twobody_10=np.mean([r["mse_twobody_10"] for r in seed_results]),
            mse_twobody_25=np.mean([r["mse_twobody_25"] for r in seed_results]),
            mse_twobody_50=np.mean([r["mse_twobody_50"] for r in seed_results]),
            improvement_10=np.mean([r["improvement_10"] for r in seed_results]),
            improvement_25=np.mean([r["improvement_25"] for r in seed_results]),
            improvement_50=np.mean([r["improvement_50"] for r in seed_results]),
            energy_drift_hgnn=np.mean([r["energy_drift_hgnn"] for r in seed_results]),
            energy_drift_twobody=np.mean([r["energy_drift_twobody"] for r in seed_results]),
            rare_regime_improvement=np.mean([r["rare_regime_improvement"] for r in seed_results]),
        )
        
        all_results.append(avg_result)
        
        print(f"\n  SUMMARY for ε={eps}:")
        print(f"    Mean |Φ|: {avg_result.mean_phi_magnitude:.6f}")
        print(f"    Improvement @10: {avg_result.improvement_10:.1f}%")
        print(f"    Improvement @25: {avg_result.improvement_25:.1f}%")
        print(f"    Improvement @50: {avg_result.improvement_50:.1f}%")
        print(f"    Rare-regime: {avg_result.rare_regime_improvement:.1f}%")
    
    # Save results
    results_file = Path(save_dir) / "sweep_results.json"
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SCALING LAW SUMMARY")
    print("=" * 70)
    print(f"{'ε':>8} {'|Φ|':>10} {'Imp@10':>10} {'Imp@25':>10} {'Imp@50':>10} {'Rare':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r.epsilon:>8.0f} {r.mean_phi_magnitude:>10.6f} "
              f"{r.improvement_10:>9.1f}% {r.improvement_25:>9.1f}% "
              f"{r.improvement_50:>9.1f}% {r.rare_regime_improvement:>9.1f}%")
    
    return all_results


if __name__ == "__main__":
    results = run_epsilon_sweep(
        epsilons=[1, 5, 10, 25, 50, 100, 200],
        n_seeds=3,
    )
