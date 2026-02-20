"""
Backward-Error Aligned Training for H-GNN.

Key innovations:
1. dt²-structured Φ: H = H_base + dt² * (1/N) * Σ Φ_tilde
2. Energy drift regularizer: (H(q_T, p_T) - H(q_0, p_0))²
3. Dual-dt training: train with dt=60 and dt=30 simultaneously
4. Zero-initialized final layer (starts exactly at base Hamiltonian)

This replaces:
- Manual λ tuning (dt² scaling handles it)
- Edge normalization (1/N scaling now)
- Energy value matching (drift is what matters)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from h_gnn.models.hgnn import HGNN
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.models.hamiltonian import TwoBodyHamiltonian
from h_gnn.integrators.symplectic import StormerVerlet


def generate_random_ic(n_particles: int, mu: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random initial conditions (circular orbits + perturbation)."""
    rng = np.random.default_rng(seed)
    
    q_list = []
    p_list = []
    
    for i in range(n_particles):
        # Random orbital radius
        r = rng.uniform(7000, 42000)
        theta = rng.uniform(0, 2 * np.pi)
        
        # Position
        q = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
        
        # Circular velocity + small perturbation
        v_circ = np.sqrt(mu / r)
        v_perturb = rng.uniform(-0.05, 0.05)
        v = (v_circ + v_perturb) * np.array([-np.sin(theta), np.cos(theta), 0.0])
        p = v  # unit mass
        
        q_list.append(q)
        p_list.append(p)
    
    q = torch.tensor(np.stack(q_list), dtype=torch.float64)
    p = torch.tensor(np.stack(p_list), dtype=torch.float64)
    
    return q, p


def dual_dt_trajectory_loss(
    model: HGNN,
    gt_hamiltonian: MultiBodyHamiltonian,
    q0: torch.Tensor,
    p0: torch.Tensor,
    edge_index: torch.Tensor,
    dt_primary: float = 60.0,
    dt_secondary: float = 30.0,
    horizons: List[int] = [5, 10, 20, 50],
    weights: List[float] = [0.2, 0.3, 0.3, 0.2],
) -> torch.Tensor:
    """
    Dual-dt trajectory loss for backward-error alignment.
    
    Train with two timesteps to force the learned Φ to be dt-robust.
    The model's dt²-structured correction automatically learns the right scaling.
    
    Note: To avoid in-place modifications breaking autograd, we create fresh
    integrators for each dt value instead of modifying model.integrator.dt.
    
    Args:
        model: H-GNN model (must have use_dt_scaling=True)
        gt_hamiltonian: Ground truth multi-body Hamiltonian
        q0, p0: Initial conditions
        edge_index: Graph connectivity
        dt_primary: Primary timestep (default 60s)
        dt_secondary: Secondary timestep (default 30s)
        horizons: Multi-scale horizons (in primary dt units)
        weights: Multi-scale weights
    """
    total_loss = 0.0
    
    # Primary dt loss (use model's default integrator)
    for T, w in zip(horizons, weights):
        # Ground truth with primary dt
        gt_integrator = StormerVerlet(dt=dt_primary, create_graph=False)
        with torch.no_grad():
            q_true, p_true = q0.double().clone(), p0.double().clone()
            for _ in range(T):
                q_true, p_true = gt_integrator.step(
                    q_true, p_true, gt_hamiltonian, edge_index
                )
        
        # Model prediction (uses model.integrator with dt_primary)
        q_pred, p_pred = model.forward(
            q0.float(), p0.float(), n_steps=T, return_trajectory=False
        )
        
        loss_T = torch.mean(
            (q_pred - q_true.float())**2 + (p_pred - p_true.float())**2
        )
        total_loss = total_loss + w * loss_T
    
    # Secondary dt loss (manual rollout to avoid modifying model.integrator)
    scale = int(dt_primary / dt_secondary)
    scaled_horizons = [T * scale for T in horizons]
    
    # Create fresh integrator for secondary dt rollout
    secondary_integrator = StormerVerlet(dt=dt_secondary, create_graph=True)
    
    for T, w in zip(scaled_horizons, weights):
        # Ground truth with secondary dt
        gt_integrator = StormerVerlet(dt=dt_secondary, create_graph=False)
        with torch.no_grad():
            q_true, p_true = q0.double().clone(), p0.double().clone()
            for _ in range(T):
                q_true, p_true = gt_integrator.step(
                    q_true, p_true, gt_hamiltonian, edge_index
                )
        
        # Manual rollout with secondary dt using model's hamiltonian
        q_pred, p_pred = q0.float(), p0.float()
        for _ in range(T):
            q_pred, p_pred = secondary_integrator.step(
                q_pred, p_pred, model.hamiltonian, edge_index
            )
        
        loss_T = torch.mean(
            (q_pred - q_true.float())**2 + (p_pred - p_true.float())**2
        )
        # Secondary dt has lower weight
        total_loss = total_loss + 0.5 * w * loss_T
    
    return total_loss


def energy_drift_regularizer(
    model: HGNN,
    gt_hamiltonian: MultiBodyHamiltonian,
    q0: torch.Tensor,
    p0: torch.Tensor,
    edge_index: torch.Tensor,
    T: int = 50,
) -> torch.Tensor:
    """
    Energy drift regularization: penalize drift in learned Hamiltonian.
    
    loss = (H_model(q_T, p_T) - H_model(q_0, p_0))²
    
    This targets Hamiltonian conservation (the essence of symplectic methods)
    rather than trying to match the exact value of H_gt.
    
    Args:
        model: H-GNN model
        gt_hamiltonian: Ground truth (only used for ground truth trajectory)
        q0, p0: Initial conditions
        edge_index: Graph connectivity
        T: Horizon for measuring drift
    """
    integrator = StormerVerlet(dt=model.integrator.dt, create_graph=False)
    
    # Evolve using ground truth to get terminal state
    with torch.no_grad():
        q_T, p_T = q0.double().clone(), p0.double().clone()
        for _ in range(T):
            q_T, p_T = integrator.step(q_T, p_T, gt_hamiltonian, edge_index)
    
    # Compute learned Hamiltonian at both endpoints
    H_0 = model.hamiltonian(q0.float(), p0.float(), edge_index)
    H_T = model.hamiltonian(q_T.float(), p_T.float(), edge_index)
    
    # Energy drift (should be small for symplectic evolution + accurate H)
    drift = H_T - H_0
    
    # Normalize by initial energy scale
    energy_scale = H_0.abs().detach() + 1e-8
    loss = (drift / energy_scale) ** 2
    
    return loss


def train_backward_error_aligned(config: Dict) -> HGNN:
    """
    Train H-GNN with backward-error aligned objective.
    
    Key differences from multiscale_training:
    1. use_dt_scaling=True: dt²-structured correction
    2. energy_drift_regularizer instead of energy_consistency_loss
    3. dual_dt_trajectory_loss for dt-robust learning
    """
    print("=" * 70)
    print("Backward-Error Aligned Training")
    print("=" * 70)
    
    # Config
    mu = config.get("mu", 398600.4418)
    dt_primary = config.get("dt_primary", 60.0)
    dt_secondary = config.get("dt_secondary", 30.0)
    n_particles = config.get("n_particles", 5)
    n_epochs = config.get("n_epochs", 300)
    lr = config.get("lr", 5e-5)
    epsilon = config.get("epsilon", 100.0)
    interaction_radius = config.get("interaction_radius", 20000.0)
    batch_size = config.get("batch_size", 10)
    
    # Multi-scale horizons and weights
    horizons = config.get("horizons", [5, 10, 20, 50])
    weights = config.get("weights", [0.2, 0.3, 0.3, 0.2])
    
    # Energy drift regularization weight
    lambda_drift = config.get("lambda_drift", 0.1)
    
    # Whether to use dual-dt training
    use_dual_dt = config.get("use_dual_dt", True)
    
    print(f"Config:")
    print(f"  epsilon = {epsilon}")
    print(f"  dt_primary = {dt_primary}, dt_secondary = {dt_secondary}")
    print(f"  use_dual_dt = {use_dual_dt}")
    print(f"  lambda_drift = {lambda_drift}")
    print(f"  Horizons: {horizons}, Weights: {weights}")
    print(f"  Particles: {n_particles}, Epochs: {n_epochs}")
    
    # Ground truth
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    
    # Edge index
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Create model with dt²-structured correction
    # Note: use_dt_scaling=True removes need for manual lambda tuning
    model = HGNN(
        mu=mu,
        interaction_strength=1.0,  # Ignored when use_dt_scaling=True
        interaction_radius=interaction_radius,
        hidden_dims=[64, 64, 32],
        integrator_type="stormer_verlet",
        dt=dt_primary,
        create_graph=True,
        softplus_output=False,
        normalize_features=True,
        init_gain=1.0,  # Combined with zero-init final layer
        use_dt_scaling=True,  # dt²-structured Φ
    ).float()
    
    print(f"\nModel config:")
    print(f"  use_dt_scaling = {model.hamiltonian.use_dt_scaling}")
    print(f"  Edges: {edge_index.shape[1]}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=30, factor=0.5, min_lr=1e-6
    )
    
    print(f"\nTraining...")
    print("-" * 70)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        
        total_traj_loss = 0.0
        total_drift_loss = 0.0
        
        for batch_idx in range(batch_size):
            seed = batch_idx + epoch * batch_size
            q0, p0 = generate_random_ic(n_particles, mu, seed)
            
            # Trajectory loss (dual-dt or single-dt)
            if use_dual_dt:
                traj_loss = dual_dt_trajectory_loss(
                    model, gt_hamiltonian,
                    q0, p0, edge_index,
                    dt_primary=dt_primary,
                    dt_secondary=dt_secondary,
                    horizons=horizons,
                    weights=weights,
                )
            else:
                # Single-dt trajectory loss (fallback)
                traj_loss = _single_dt_trajectory_loss(
                    model, gt_hamiltonian,
                    q0, p0, edge_index,
                    dt=dt_primary,
                    horizons=horizons,
                    weights=weights,
                )
            
            # Energy drift regularizer
            drift_loss = energy_drift_regularizer(
                model, gt_hamiltonian, q0, p0, edge_index, T=50
            )
            
            # Alpha regularization (prevents explosion, reduces seed variance)
            alpha_reg = 1e-4 * model.hamiltonian.alpha.pow(2)
            
            # Combined loss
            loss = traj_loss + lambda_drift * drift_loss + alpha_reg
            loss.backward()
            
            total_traj_loss += traj_loss.item()
            total_drift_loss += drift_loss.item()
        
        avg_traj = total_traj_loss / batch_size
        avg_drift = total_drift_loss / batch_size
        avg_total = avg_traj + lambda_drift * avg_drift
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(avg_total)
        
        if avg_total < best_loss:
            best_loss = avg_total
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if epoch % 30 == 0 or epoch == n_epochs - 1:
            # Check Phi output
            with torch.no_grad():
                q_test, p_test = generate_random_ic(n_particles, mu, 9999)
                q_i = q_test.float()[edge_index[0]]
                p_i = p_test.float()[edge_index[0]]
                q_j = q_test.float()[edge_index[1]]
                p_j = p_test.float()[edge_index[1]]
                phi_out = model.hamiltonian.Phi(q_i, p_i, q_j, p_j)
            
            print(f"Epoch {epoch:3d}: Traj={avg_traj:.4e}, Drift={avg_drift:.4e}, "
                  f"|Phi|={phi_out.abs().mean().item():.4f}")
    
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print("-" * 70)
    print(f"Training complete. Best loss: {best_loss:.6e}")
    
    # Save
    torch.save(model.state_dict(), "hgnn_backward_error.pt")
    print("Model saved to hgnn_backward_error.pt")
    
    return model


def _single_dt_trajectory_loss(
    model: HGNN,
    gt_hamiltonian: MultiBodyHamiltonian,
    q0: torch.Tensor,
    p0: torch.Tensor,
    edge_index: torch.Tensor,
    dt: float,
    horizons: List[int],
    weights: List[float],
) -> torch.Tensor:
    """Single-dt trajectory loss (fallback for comparison)."""
    total_loss = 0.0
    integrator = StormerVerlet(dt=dt, create_graph=False)
    
    for T, w in zip(horizons, weights):
        with torch.no_grad():
            q_true, p_true = q0.double().clone(), p0.double().clone()
            for _ in range(T):
                q_true, p_true = integrator.step(
                    q_true, p_true, gt_hamiltonian, edge_index
                )
        
        q_pred, p_pred = model.forward(
            q0.float(), p0.float(), n_steps=T, return_trajectory=False
        )
        
        loss_T = torch.mean(
            (q_pred - q_true.float())**2 + (p_pred - p_true.float())**2
        )
        
        total_loss = total_loss + w * loss_T
    
    return total_loss


def evaluate_backward_error_model(model: HGNN, config: Dict) -> Dict:
    """Evaluate backward-error aligned model."""
    print("\n" + "=" * 70)
    print("Evaluation: Backward-Error Aligned H-GNN")
    print("=" * 70)
    
    mu = config.get("mu", 398600.4418)
    dt = config.get("dt_primary", 60.0)
    epsilon = config.get("epsilon", 100.0)
    n_particles = config.get("n_particles", 5)
    n_eval = config.get("n_eval", 20)
    
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    twobody_H = TwoBodyHamiltonian(mu=mu)
    integrator = StormerVerlet(dt=dt, create_graph=False)
    
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    results = {}
    horizons = [5, 10, 25, 50, 100, 200]
    
    model.eval()
    model.integrator.dt.fill_(dt)
    
    print(f"{'Horizon':>8} {'H-GNN MSE':>12} {'Two-body MSE':>14} {'Improvement':>12} {'E_drift':>12}")
    print("-" * 70)
    
    for T in horizons:
        mse_hgnn = []
        mse_tb = []
        energy_drift = []
        
        for seed in range(n_eval):
            q0, p0 = generate_random_ic(n_particles, mu, seed + 1000)
            
            with torch.no_grad():
                # Ground truth
                q_gt, p_gt = q0.clone(), p0.clone()
                for _ in range(T):
                    q_gt, p_gt = integrator.step(q_gt, p_gt, gt_hamiltonian, edge_index)
                
                # Two-body baseline
                q_tb, p_tb = q0.clone(), p0.clone()
                for _ in range(T):
                    q_tb, p_tb = integrator.step(q_tb, p_tb, twobody_H, None)
                
                # H-GNN
                q_hgnn, p_hgnn = model.forward(
                    q0.float(), p0.float(), n_steps=T, return_trajectory=False
                )
                
                # Energy drift
                H_0 = model.hamiltonian(q0.float(), p0.float(), edge_index)
                H_T = model.hamiltonian(q_hgnn.float(), p_hgnn.float(), edge_index)
                
                mse_hgnn.append(torch.mean((q_hgnn - q_gt.float())**2).item())
                mse_tb.append(torch.mean((q_tb.float() - q_gt.float())**2).item())
                energy_drift.append(((H_T - H_0) / (H_0.abs() + 1e-8)).abs().item())
        
        avg_hgnn = np.mean(mse_hgnn)
        avg_tb = np.mean(mse_tb)
        avg_drift = np.mean(energy_drift)
        improvement = (avg_tb - avg_hgnn) / avg_tb * 100
        
        results[T] = {
            "mse_hgnn": avg_hgnn,
            "mse_twobody": avg_tb,
            "improvement": improvement,
            "energy_drift": avg_drift,
        }
        
        print(f"{T:>8} {avg_hgnn:>12.4e} {avg_tb:>14.4e} {improvement:>11.1f}% {avg_drift:>12.4e}")
    
    return results


def run_n_scaling_test(config: Dict, n_values: List[int] = [10, 25], n_seeds: int = 3) -> Dict:
    """
    N-scaling test: train at multiple particle counts.
    
    With 1/N normalization, improvement should be stable across N.
    """
    print("\n" + "=" * 70)
    print("N-Scaling Test (Backward-Error Aligned)")
    print("=" * 70)
    
    results = {}
    
    for n in n_values:
        print(f"\n--- N = {n} ---")
        improvements = []
        
        for seed in range(n_seeds):
            print(f"  Seed {seed + 1}/{n_seeds}")
            
            # Set seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Update config
            cfg = config.copy()
            cfg["n_particles"] = n
            cfg["n_epochs"] = 150  # Faster for testing
            cfg["batch_size"] = 8
            
            # Train
            model = train_backward_error_aligned(cfg)
            
            # Evaluate at T=50
            eval_results = evaluate_backward_error_model(model, cfg)
            improvement = eval_results.get(50, {}).get("improvement", 0.0)
            improvements.append(improvement)
            
            print(f"    T=50 improvement: {improvement:.1f}%")
        
        results[n] = {
            "mean": np.mean(improvements),
            "std": np.std(improvements),
            "values": improvements,
        }
        
        print(f"\nN={n}: {np.mean(improvements):.1f}% ± {np.std(improvements):.1f}%")
    
    print("\n" + "=" * 70)
    print("N-Scaling Summary")
    print("-" * 70)
    for n, r in results.items():
        print(f"N={n}: {r['mean']:.1f}% ± {r['std']:.1f}%")
    
    return results


def run_dt_scaling_test(config: Dict) -> Dict:
    """
    dt-scaling test: verify MSE ~ dt² behavior.
    
    Train model at dt=60, evaluate at dt=60 and dt=30.
    Expected: MSE(dt=60) / MSE(dt=30) ≈ 4 (if backward-error aligned).
    """
    print("\n" + "=" * 70)
    print("dt-Scaling Test (Backward-Error Aligned)")
    print("=" * 70)
    
    # Train at dt=60
    cfg = config.copy()
    cfg["dt_primary"] = 60.0
    cfg["dt_secondary"] = 30.0
    cfg["n_epochs"] = 200
    
    model = train_backward_error_aligned(cfg)
    
    # Evaluate at both dt values
    dt_values = [60.0, 30.0]
    results = {}
    
    for dt in dt_values:
        cfg_eval = cfg.copy()
        cfg_eval["dt_primary"] = dt
        model.integrator.dt.fill_(dt)
        
        eval_results = evaluate_backward_error_model(model, cfg_eval)
        results[dt] = eval_results
    
    # Compute ratio at T=50 steps (equal physical time)
    mse_60 = results[60.0][50]["mse_hgnn"]
    mse_30 = results[30.0][100]["mse_hgnn"]  # 100 steps at dt=30 = 50 steps at dt=60
    
    ratio = mse_60 / mse_30 if mse_30 > 0 else float('inf')
    
    print(f"\ndt-Scaling Result:")
    print(f"  MSE(dt=60, T=50): {mse_60:.4e}")
    print(f"  MSE(dt=30, T=100): {mse_30:.4e}")
    print(f"  Ratio: {ratio:.2f} (expected ~4 for dt² scaling)")
    
    return {"ratio": ratio, "mse_60": mse_60, "mse_30": mse_30}


def run_t100_multiseed_test(config: Dict, n_seeds: int = 3) -> Dict:
    """
    T=100 multi-seed validation.
    
    Expected: positive improvement with variance < 15%.
    """
    print("\n" + "=" * 70)
    print(f"T=100 Multi-Seed Validation ({n_seeds} seeds)")
    print("=" * 70)
    
    improvements = []
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = train_backward_error_aligned(config)
        eval_results = evaluate_backward_error_model(model, config)
        
        improvement = eval_results.get(100, {}).get("improvement", 0.0)
        improvements.append(improvement)
        print(f"  T=100 improvement: {improvement:.1f}%")
    
    mean_imp = np.mean(improvements)
    std_imp = np.std(improvements)
    
    print("\n" + "=" * 70)
    print(f"T=100 Multi-Seed Result:")
    print(f"  Mean: {mean_imp:.1f}%")
    print(f"  Std: {std_imp:.1f}%")
    print(f"  All positive: {all(x > 0 for x in improvements)}")
    print(f"  Variance acceptable: {std_imp < 15}")
    
    return {
        "mean": mean_imp,
        "std": std_imp,
        "values": improvements,
        "all_positive": all(x > 0 for x in improvements),
    }


if __name__ == "__main__":
    # Default config for backward-error aligned training
    config = {
        "mu": 398600.4418,
        "dt_primary": 60.0,
        "dt_secondary": 30.0,
        "n_particles": 5,
        "n_epochs": 300,
        "lr": 5e-5,
        "epsilon": 100.0,
        "interaction_radius": 20000.0,
        "batch_size": 10,
        "horizons": [5, 10, 20, 50],
        "weights": [0.2, 0.3, 0.3, 0.2],
        "lambda_drift": 0.1,
        "use_dual_dt": True,
        "n_eval": 20,
    }
    
    print("=" * 70)
    print("Backward-Error Aligned H-GNN Training")
    print("=" * 70)
    
    # Train and evaluate
    model = train_backward_error_aligned(config)
    results = evaluate_backward_error_model(model, config)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
