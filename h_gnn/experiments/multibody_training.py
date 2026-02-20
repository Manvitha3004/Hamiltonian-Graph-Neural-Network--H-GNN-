"""
Multi-Body Training Experiment.

Tests if H-GNN can learn pairwise gravitational interactions
that are NOT in the base Hamiltonian.

Setup:
- Ground truth: Two-body + inter-satellite gravity (ε > 0)
- Model base: Pure two-body
- Goal: Φ should learn to approximate inter-satellite gravity

Training uses trajectory-based loss which is more stable than
gradient matching for small corrections.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

from h_gnn.models import HGNN, TwoBodyHamiltonian
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet
from h_gnn.data import SyntheticCatalog


def generate_random_ic(n_particles: int, mu: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random initial conditions for multi-body system."""
    np.random.seed(seed)
    r = 7000 + 500 * np.random.randn(n_particles)
    theta = np.linspace(0, 2*np.pi, n_particles, endpoint=False) + 0.1*np.random.randn(n_particles)
    
    q = torch.tensor([
        [r[i]*np.cos(theta[i]), r[i]*np.sin(theta[i]), 50*np.random.randn()] 
        for i in range(n_particles)
    ], dtype=torch.float64)
    
    v = np.sqrt(mu / r)
    p = torch.tensor([
        [v[i]*(-np.sin(theta[i])), v[i]*np.cos(theta[i]), 0.1*np.random.randn()] 
        for i in range(n_particles)
    ], dtype=torch.float64)
    
    return q, p


def train_with_multibody_ground_truth(config: Dict) -> HGNN:
    """
    Train H-GNN to learn inter-satellite gravitational corrections.
    
    Uses trajectory-based loss: minimize error between model prediction
    and ground truth multi-body dynamics over short rollouts.
    
    Key hyperparameters for stable training:
    - Small interaction_strength (λ=0.1): prevents gradient explosion
    - Low learning rate (1e-4): stable convergence
    - Aggressive gradient clipping (0.1): prevents divergence
    - Multi-step rollout (5 steps): learns temporal dynamics
    - Mini-batch training: improves generalization
    """
    print("=" * 60)
    print("Multi-Body H-GNN Training (Trajectory Loss)")
    print("=" * 60)
    
    # Config
    mu = config.get("mu", 398600.4418)
    dt = config.get("dt", 60.0)
    n_particles = config.get("n_particles", 5)
    n_epochs = config.get("n_epochs", 200)
    lr = config.get("lr", 1e-4)
    epsilon = config.get("epsilon", 100.0)
    interaction_radius = config.get("interaction_radius", 20000.0)
    rollout_steps = config.get("rollout_steps", 5)
    batch_size = config.get("batch_size", 10)
    
    print(f"Inter-satellite coupling ε = {epsilon}")
    print(f"Particles: {n_particles}, Rollout: {rollout_steps} steps")
    print(f"Learning rate: {lr}, Batch size: {batch_size}")
    
    # Ground truth Hamiltonian
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    base_integrator = StormerVerlet(dt=dt, create_graph=False)
    
    # Build edge index for n_particles
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Model - CRITICAL: small lambda and proper normalization
    model = HGNN(
        mu=mu,
        interaction_strength=0.1,  # Small lambda for stability
        interaction_radius=interaction_radius,
        hidden_dims=[64, 64, 32],
        integrator_type="stormer_verlet",
        dt=dt,
        create_graph=True,
        softplus_output=False,  # Allow negative corrections
        normalize_features=True,  # Normalize input features
    ).float()
    
    print(f"Model λ = {model.hamiltonian.lambda_.item():.2f}")
    print(f"Edges: {edge_index.shape[1]}")
    
    # Optimizer with lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=30, factor=0.5, min_lr=1e-6
    )
    
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 40)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        
        total_loss = 0.0
        for batch_idx in range(batch_size):
            seed = batch_idx + epoch * batch_size
            q0, p0 = generate_random_ic(n_particles, mu, seed)
            
            # Ground truth trajectory
            with torch.no_grad():
                q_true, p_true = q0.clone(), p0.clone()
                for _ in range(rollout_steps):
                    q_true, p_true = base_integrator.step(
                        q_true, p_true, gt_hamiltonian, edge_index
                    )
            
            # Model prediction
            q_pred, p_pred = model.forward(
                q0.float(), p0.float(), n_steps=rollout_steps, return_trajectory=False
            )
            
            # Trajectory loss
            loss = torch.mean(
                (q_pred - q_true.float())**2 + (p_pred - p_true.float())**2
            )
            loss.backward()
            total_loss += loss.item()
        
        avg_loss = total_loss / batch_size
        
        # Aggressive gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if epoch % 40 == 0 or epoch == n_epochs - 1:
            # Check Phi output magnitude
            with torch.no_grad():
                q_test, p_test = generate_random_ic(n_particles, mu, 9999)
                q_i = q_test.float()[edge_index[0]]
                p_i = p_test.float()[edge_index[0]]
                q_j = q_test.float()[edge_index[1]]
                p_j = p_test.float()[edge_index[1]]
                phi_out = model.hamiltonian.Phi(q_i, p_i, q_j, p_j)
            
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.6e}, "
                  f"|Φ|={phi_out.abs().mean().item():.6f}")
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print("-" * 40)
    print(f"Training complete. Best loss: {best_loss:.6e}")
    
    # Save model
    torch.save(model.state_dict(), "hgnn_multibody.pt")
    print("Model saved to hgnn_multibody.pt")
    
    return model


def evaluate_multibody_model(model: HGNN, config: Dict) -> Dict:
    """
    Evaluate trained model vs two-body baseline over multiple time horizons.
    """
    print("\n" + "=" * 60)
    print("Evaluation: H-GNN vs Two-Body Baseline")
    print("=" * 60)
    
    mu = config.get("mu", 398600.4418)
    dt = config.get("dt", 60.0)
    epsilon = config.get("epsilon", 100.0)
    n_particles = config.get("n_particles", 5)
    n_eval = config.get("n_eval", 20)
    
    # Ground truth
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    twobody_H = TwoBodyHamiltonian(mu=mu)
    integrator = StormerVerlet(dt=dt, create_graph=False)
    
    # Build edge index
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    results = {}
    horizons = [10, 50, 100, 200]
    
    model.eval()
    
    for T in horizons:
        mse_hgnn = []
        mse_tb = []
        
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
                
                # H-GNN prediction
                q_hgnn, p_hgnn = model(q0.float(), p0.float(), n_steps=T)
            
            mse_hgnn.append(torch.mean((q_hgnn - q_gt.float())**2).item())
            mse_tb.append(torch.mean((q_tb.float() - q_gt.float())**2).item())
        
        avg_hgnn = np.mean(mse_hgnn)
        avg_tb = np.mean(mse_tb)
        improvement = 100 * (avg_tb - avg_hgnn) / avg_tb if avg_tb > 0 else 0
        
        results[f"T{T}_mse_hgnn"] = avg_hgnn
        results[f"T{T}_mse_tb"] = avg_tb
        results[f"T{T}_improvement"] = improvement
        
        print(f"T={T:3d}: H-GNN={avg_hgnn:.2e}, Two-body={avg_tb:.2e}, Improvement={improvement:+.1f}%")
    
    # λ=0 ablation at T=50
    print("\nAblation study (λ=0):")
    model.disable_interaction()
    mse_ablation = []
    for seed in range(n_eval):
        q0, p0 = generate_random_ic(n_particles, mu, seed + 1000)
        with torch.no_grad():
            q_gt, p_gt = q0.clone(), p0.clone()
            for _ in range(50):
                q_gt, p_gt = integrator.step(q_gt, p_gt, gt_hamiltonian, edge_index)
            q_abl, p_abl = model(q0.float(), p0.float(), n_steps=50)
        mse_ablation.append(torch.mean((q_abl - q_gt.float())**2).item())
    model.enable_interaction()
    
    avg_ablation = np.mean(mse_ablation)
    avg_hgnn_50 = results["T50_mse_hgnn"]
    
    print(f"T=50: λ=0 MSE={avg_ablation:.2e}, λ>0 MSE={avg_hgnn_50:.2e}")
    if avg_hgnn_50 < avg_ablation:
        print("✓ Interaction term helps (λ>0 better than λ=0)")
    else:
        print("✗ Interaction term does not help")
    
    results["T50_mse_ablation"] = avg_ablation
    
    return results


if __name__ == "__main__":
    config = {
        "mu": 398600.4418,
        "dt": 60.0,
        "n_particles": 5,
        "n_epochs": 200,
        "lr": 1e-4,
        "epsilon": 100.0,
        "rollout_steps": 5,
        "batch_size": 10,
        "interaction_radius": 20000.0,
        "n_eval": 20,
    }
    
    model = train_with_multibody_ground_truth(config)
    results = evaluate_multibody_model(model, config)
