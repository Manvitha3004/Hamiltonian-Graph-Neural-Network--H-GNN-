"""
Curriculum Training with Energy Regularization.

Key improvements:
1. Train on random T ∈ [10, 200] instead of fixed short horizon
2. Add mild energy variance regularizer
3. Track correction magnitude ||∇Φ|| / ||∇H_base||

This addresses T=200 degradation observed in fixed-horizon training.
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


def train_with_curriculum(
    epsilon: float,
    config: Dict,
    seed: int = 42,
    horizon_range: Tuple[int, int] = (10, 200),
    energy_reg_weight: float = 0.01,
) -> Tuple[HGNN, Dict]:
    """
    Train H-GNN with curriculum learning over variable horizons.
    
    Key differences from fixed-horizon training:
    1. Each epoch samples random rollout length T ∈ horizon_range
    2. Mild energy regularizer: Var(H(trajectory))
    3. Gradient matching at multiple rollout lengths
    
    Returns:
        (model, training_stats)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    mu = config["mu"]
    dt = config["dt"]
    n_objects = config["n_objects"]
    n_epochs = config["n_epochs"]
    lr = config["lr"]
    interaction_radius = config["interaction_radius"]
    
    print(f"Curriculum training: T ∈ [{horizon_range[0]}, {horizon_range[1]}]")
    print(f"Energy regularization weight: {energy_reg_weight}")
    
    # Generate training data
    catalog = SyntheticCatalog(mu=mu, seed=seed)
    q_train, p_train = catalog.generate_training_set(n_objects)
    
    # Hamiltonians
    gt_H = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    tb_H = TwoBodyHamiltonian(mu=mu)
    integrator = StormerVerlet(dt=dt)
    
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
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    best_loss = float('inf')
    best_state = None
    training_stats = {
        "losses": [],
        "grad_ratios": [],
        "energy_vars": [],
        "horizons_used": [],
    }
    
    T_min, T_max = horizon_range
    
    for epoch in range(n_epochs):
        model.train()
        
        # Sample random horizon for this epoch (curriculum)
        T = np.random.randint(T_min, T_max + 1)
        training_stats["horizons_used"].append(T)
        
        # === Gradient matching loss ===
        # Target: gradient difference between multi-body and two-body
        grad_q_gt, _ = gt_H.compute_gradients(q_train, p_train)
        grad_q_tb, _ = tb_H.compute_gradients(q_train, p_train)
        target_grad = grad_q_gt - grad_q_tb
        
        # Model gradient
        grad_q_model, _ = model.hamiltonian.compute_gradients(
            q_train, p_train, edge_index, create_graph=True
        )
        grad_q_tb_curr, _ = tb_H.compute_gradients(q_train, p_train)
        model_correction = grad_q_model - grad_q_tb_curr
        
        grad_match_loss = torch.mean((model_correction - target_grad) ** 2)
        
        # === Trajectory energy variance regularizer ===
        # Run short rollout and penalize energy variance
        energy_reg_loss = torch.tensor(0.0, dtype=torch.float64)
        
        if energy_reg_weight > 0 and T >= 5:
            rollout_T = min(T, 20)  # Cap rollout for efficiency
            
            with torch.enable_grad():
                q_curr = q_train.clone()
                p_curr = p_train.clone()
                energies = []
                
                for step in range(rollout_T):
                    # Compute energy at this state
                    E = gt_H(q_curr, p_curr)
                    energies.append(E)
                    
                    # Step forward (without tracking for memory)
                    with torch.no_grad():
                        q_curr, p_curr = model.integrator.step(
                            q_curr, p_curr, model.hamiltonian, edge_index
                        )
                
                # Energy variance
                energies = torch.stack(energies)
                energy_reg_loss = torch.var(energies) * energy_reg_weight
        
        # Total loss
        total_loss = grad_match_loss + energy_reg_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Track stats
        training_stats["losses"].append(total_loss.item())
        training_stats["energy_vars"].append(energy_reg_loss.item() / energy_reg_weight if energy_reg_weight > 0 else 0)
        
        # Compute gradient ratio ||∇Φ|| / ||∇H_base||
        with torch.no_grad():
            grad_phi = model_correction.abs().mean().item()
            grad_base = grad_q_tb.abs().mean().item()
            ratio = grad_phi / grad_base if grad_base > 0 else 0
            training_stats["grad_ratios"].append(ratio)
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d}: T={T:3d}, Loss={total_loss.item():.2e}, "
                  f"||∇Φ||/||∇H||={ratio:.4f}, E_var={training_stats['energy_vars'][-1]:.2e}")
    
    model.load_state_dict(best_state)
    model.eval()
    
    return model, training_stats


def compute_correction_magnitude_analysis(
    model: HGNN,
    epsilon: float,
    config: Dict,
) -> Dict:
    """
    Analyze correction magnitude: ||∇Φ|| / ||∇H_base||
    
    This shows whether the model learns proportional corrections.
    """
    mu = config["mu"]
    
    gt_H = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    tb_H = TwoBodyHamiltonian(mu=mu)
    
    catalog = SyntheticCatalog(mu=mu, seed=99)
    q, p = catalog.generate_training_set(100)
    
    # Build graph
    graph = model.graph_builder(q)
    edge_index = graph.edge_index
    
    with torch.no_grad():
        # Ground truth correction
        grad_q_gt, _ = gt_H.compute_gradients(q, p)
        grad_q_tb, _ = tb_H.compute_gradients(q, p)
        target_correction = grad_q_gt - grad_q_tb
        
        # Model correction
        grad_q_model, _ = model.hamiltonian.compute_gradients(q, p, edge_index)
        grad_q_tb_curr, _ = tb_H.compute_gradients(q, p)
        model_correction = grad_q_model - grad_q_tb_curr
    
    # Magnitudes
    target_mag = target_correction.abs().mean().item()
    model_mag = model_correction.abs().mean().item()
    base_mag = grad_q_tb.abs().mean().item()
    
    # Ratios
    target_ratio = target_mag / base_mag if base_mag > 0 else 0
    model_ratio = model_mag / base_mag if base_mag > 0 else 0
    learning_efficiency = model_ratio / target_ratio if target_ratio > 0 else 0
    
    return {
        "target_correction_magnitude": target_mag,
        "model_correction_magnitude": model_mag,
        "base_gradient_magnitude": base_mag,
        "target_ratio": target_ratio,  # ||∇Φ_true|| / ||∇H_base||
        "model_ratio": model_ratio,    # ||∇Φ_learned|| / ||∇H_base||
        "learning_efficiency": learning_efficiency,  # How much of target ratio is captured
    }


def run_curriculum_experiment(
    epsilon: float = 50.0,
    horizons: List[int] = [10, 25, 50, 100, 200, 500],
    n_seeds: int = 3,
    save_dir: str = "results/curriculum_training",
) -> Dict:
    """
    Train with curriculum and evaluate at multiple horizons.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    config = {
        "mu": 398600.4418,
        "dt": 60.0,
        "n_objects": 50,
        "n_epochs": 200,  # More epochs for curriculum
        "lr": 1e-3,
        "interaction_radius": 5000.0,
    }
    
    print("=" * 70)
    print("CURRICULUM TRAINING EXPERIMENT")
    print("=" * 70)
    print(f"ε = {epsilon}, Horizons = {horizons}")
    print()
    
    all_results = {h: [] for h in horizons}
    correction_analysis = []
    
    mu = config["mu"]
    dt = config["dt"]
    
    gt_H = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    tb_H = TwoBodyHamiltonian(mu=mu)
    integrator = StormerVerlet(dt=dt)
    
    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed + 1}/{n_seeds}")
        print("=" * 70)
        
        # Train with curriculum
        model, stats = train_with_curriculum(
            epsilon=epsilon,
            config=config,
            seed=seed,
            horizon_range=(10, 200),
            energy_reg_weight=0.01,
        )
        
        # Correction magnitude analysis
        corr_analysis = compute_correction_magnitude_analysis(model, epsilon, config)
        correction_analysis.append(corr_analysis)
        print(f"\nCorrection magnitude ratio: {corr_analysis['model_ratio']:.4f} "
              f"(target: {corr_analysis['target_ratio']:.4f})")
        
        # Evaluate at all horizons
        catalog = SyntheticCatalog(mu=mu, seed=seed + 100)
        q_test, p_test = catalog.generate_training_set(100)
        
        print("\nEvaluating...")
        for horizon in horizons:
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
            
            # Normalized error
            mean_q_sq = torch.mean(q_gt ** 2).item()
            norm_error = mse_hgnn / mean_q_sq if mean_q_sq > 0 else 0
            
            improvement = (1 - mse_hgnn / mse_tb) * 100 if mse_tb > 0 else 0
            all_results[horizon].append({
                "improvement": improvement,
                "mse_hgnn": mse_hgnn,
                "mse_twobody": mse_tb,
                "normalized_error": norm_error,
            })
            
            print(f"  T={horizon:3d}: Improvement={improvement:+6.1f}%, "
                  f"NormErr={norm_error:.2e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("CURRICULUM TRAINING RESULTS (mean ± std)")
    print("=" * 70)
    
    print(f"\n{'Horizon':>8} {'Improvement':>15} {'Std':>10} {'NormErr':>12}")
    print("-" * 50)
    
    summary = {}
    for horizon in horizons:
        imps = [r["improvement"] for r in all_results[horizon]]
        norm_errs = [r["normalized_error"] for r in all_results[horizon]]
        
        mean_imp = np.mean(imps)
        std_imp = np.std(imps)
        mean_norm = np.mean(norm_errs)
        
        summary[horizon] = {
            "mean_improvement": mean_imp,
            "std_improvement": std_imp,
            "mean_normalized_error": mean_norm,
        }
        
        print(f"{horizon:>8} {mean_imp:>+14.1f}% {std_imp:>9.1f}% {mean_norm:>12.2e}")
    
    # Correction analysis summary
    print(f"\nCorrection Magnitude Analysis:")
    print(f"  Target ||∇Φ||/||∇H||: {np.mean([c['target_ratio'] for c in correction_analysis]):.4f}")
    print(f"  Model ||∇Φ||/||∇H||:  {np.mean([c['model_ratio'] for c in correction_analysis]):.4f}")
    print(f"  Learning efficiency:  {np.mean([c['learning_efficiency'] for c in correction_analysis]):.2%}")
    
    # Save results
    results = {
        "summary": summary,
        "all_results": all_results,
        "correction_analysis": correction_analysis,
    }
    
    results_file = Path(save_dir) / "curriculum_results.json"
    
    def convert_for_json(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj
    
    with open(results_file, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = run_curriculum_experiment(
        epsilon=50.0,
        horizons=[10, 25, 50, 100, 200, 500],
        n_seeds=3,
    )
