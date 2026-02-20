"""
Multi-Scale Training with Energy-Consistency Regularization.

Key improvements over curriculum training:
1. Geometric mixture of horizons: T ∈ {5, 10, 20, 50} with weights
2. Energy-consistency loss: ||H_model - H_gt||²
3. Scaled Φ reparameterization: Φ = α * Ψ where Ψ is O(1)

These changes specifically target the T≥100 degradation problem.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path

from h_gnn.models import HGNN, TwoBodyHamiltonian
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet


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


class ScaledInteractionPotential(nn.Module):
    """
    Reparameterized interaction potential: Φ = α * Ψ
    
    Where α is a fixed scale factor (e.g., 0.03) and Ψ is O(1).
    This removes scale imbalance between Φ and H_base gradients.
    """
    
    def __init__(self, base_potential: nn.Module, alpha: float = 0.03):
        super().__init__()
        self.base_potential = base_potential  # This outputs Ψ (O(1))
        self.alpha = alpha  # Fixed scale factor
        
    def forward(self, q_i, p_i, q_j, p_j):
        psi = self.base_potential(q_i, p_i, q_j, p_j)
        return self.alpha * psi


def multiscale_loss(
    model: HGNN,
    gt_hamiltonian: MultiBodyHamiltonian,
    base_integrator: StormerVerlet,
    q0: torch.Tensor,
    p0: torch.Tensor,
    edge_index: torch.Tensor,
    horizons: List[int] = [5, 10, 20, 50],
    weights: List[float] = [0.2, 0.3, 0.3, 0.2],
) -> torch.Tensor:
    """
    Geometric mixture of trajectory losses at different horizons.
    
    L = Σ w_T * L_T
    
    More weight on short T (stability), some on medium T (consistency).
    """
    total_loss = 0.0
    
    for T, w in zip(horizons, weights):
        # Ground truth trajectory
        with torch.no_grad():
            q_true, p_true = q0.double().clone(), p0.double().clone()
            for _ in range(T):
                q_true, p_true = base_integrator.step(
                    q_true, p_true, gt_hamiltonian, edge_index
                )
        
        # Model prediction
        q_pred, p_pred = model.forward(
            q0.float(), p0.float(), n_steps=T, return_trajectory=False
        )
        
        # Trajectory MSE for this horizon
        loss_T = torch.mean(
            (q_pred - q_true.float())**2 + (p_pred - p_true.float())**2
        )
        
        total_loss = total_loss + w * loss_T
    
    return total_loss


def energy_consistency_loss(
    model: HGNN,
    gt_hamiltonian: MultiBodyHamiltonian,
    q: torch.Tensor,
    p: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Energy-consistency regularization: ||H_model - H_gt||²
    
    This directly targets long-horizon drift by ensuring the learned
    Hamiltonian approximates the true one.
    """
    # Model Hamiltonian (H_base + λΦ)
    H_model = model.hamiltonian(q.float(), p.float(), edge_index)
    
    # Ground truth Hamiltonian (no edge_index - it computes all pairs internally)
    with torch.no_grad():
        H_gt = gt_hamiltonian(q.double(), p.double()).float()
    
    # Energy difference (normalized by typical energy scale)
    energy_scale = H_gt.abs() + 1e-8
    loss = ((H_model - H_gt) / energy_scale) ** 2
    
    return loss


def train_multiscale(config: Dict) -> HGNN:
    """
    Train H-GNN with multi-scale loss and energy-consistency regularization.
    """
    print("=" * 70)
    print("Multi-Scale Training with Energy-Consistency Regularization")
    print("=" * 70)
    
    # Config
    mu = config.get("mu", 398600.4418)
    dt = config.get("dt", 60.0)
    n_particles = config.get("n_particles", 5)
    n_epochs = config.get("n_epochs", 300)
    lr = config.get("lr", 1e-4)
    epsilon = config.get("epsilon", 100.0)
    interaction_radius = config.get("interaction_radius", 20000.0)
    batch_size = config.get("batch_size", 10)
    
    # Multi-scale horizons and weights
    horizons = config.get("horizons", [5, 10, 20, 50])
    weights = config.get("weights", [0.2, 0.3, 0.3, 0.2])
    
    # Energy regularization weight
    lambda_energy = config.get("lambda_energy", 0.1)
    
    # Φ scale factor (for reparameterization)
    phi_scale = config.get("phi_scale", 0.005)
    
    print(f"epsilon = {epsilon}, lambda_energy = {lambda_energy}, phi_scale = {phi_scale}")
    print(f"Horizons: {horizons}, Weights: {weights}")
    print(f"Particles: {n_particles}, Epochs: {n_epochs}")
    
    # Ground truth
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    base_integrator = StormerVerlet(dt=dt, create_graph=False)
    
    # Edge index
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Create model with scaled interaction (Φ = λ * Ψ where Ψ is O(1))
    # init_gain=1.0 makes Ψ output O(1), then λ=phi_scale scales it down
    model = HGNN(
        mu=mu,
        interaction_strength=phi_scale,  # λ scales O(1) interaction (normalized by num_edges)
        interaction_radius=interaction_radius,
        hidden_dims=[64, 64, 32],
        integrator_type="stormer_verlet",
        dt=dt,
        create_graph=True,
        softplus_output=False,
        normalize_features=True,
        init_gain=1.0,  # O(1) output before scaling by lambda
    ).float()
    
    print(f"Model lambda = {model.hamiltonian.lambda_.item():.4f}")
    print(f"Edges: {edge_index.shape[1]}")
    
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
        total_energy_loss = 0.0
        
        for batch_idx in range(batch_size):
            seed = batch_idx + epoch * batch_size
            q0, p0 = generate_random_ic(n_particles, mu, seed)
            
            # Multi-scale trajectory loss
            traj_loss = multiscale_loss(
                model, gt_hamiltonian, base_integrator,
                q0, p0, edge_index, horizons, weights
            )
            
            # Energy-consistency regularization
            energy_loss = energy_consistency_loss(
                model, gt_hamiltonian, q0, p0, edge_index
            )
            
            # Combined loss
            loss = traj_loss + lambda_energy * energy_loss
            loss.backward()
            
            total_traj_loss += traj_loss.item()
            total_energy_loss += energy_loss.item()
        
        avg_traj = total_traj_loss / batch_size
        avg_energy = total_energy_loss / batch_size
        avg_total = avg_traj + lambda_energy * avg_energy
        
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
            
            print(f"Epoch {epoch:3d}: Traj={avg_traj:.4e}, Energy={avg_energy:.4e}, "
                  f"|Φ|={phi_out.abs().mean().item():.4f}")
    
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print("-" * 70)
    print(f"Training complete. Best loss: {best_loss:.6e}")
    
    # Save
    torch.save(model.state_dict(), "hgnn_multiscale.pt")
    print("Model saved to hgnn_multiscale.pt")
    
    return model


def evaluate_multiscale(model: HGNN, config: Dict) -> Dict:
    """
    Evaluate at multiple horizons including T=100 and T=200.
    """
    print("\n" + "=" * 70)
    print("Evaluation: Multi-Scale Trained H-GNN")
    print("=" * 70)
    
    mu = config.get("mu", 398600.4418)
    dt = config.get("dt", 60.0)
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
    
    print(f"{'Horizon':>8} {'H-GNN MSE':>12} {'Two-body MSE':>14} {'Improvement':>12} {'E_drift':>12}")
    print("-" * 70)
    
    for T in horizons:
        mse_hgnn = []
        mse_tb = []
        energy_drift = []
        
        for seed in range(n_eval):
            q0, p0 = generate_random_ic(n_particles, mu, seed + 1000)
            
            with torch.no_grad():
                # Initial energy (no edge_index for MultiBodyHamiltonian)
                H0 = gt_hamiltonian(q0, p0).item()
                
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
                
                # H-GNN energy at end
                H_end = model.hamiltonian(q_hgnn, p_hgnn, edge_index).item()
            
            mse_hgnn.append(torch.mean((q_hgnn - q_gt.float())**2).item())
            mse_tb.append(torch.mean((q_tb.float() - q_gt.float())**2).item())
            energy_drift.append(abs(H_end - H0) / (abs(H0) + 1e-8))
        
        avg_hgnn = np.mean(mse_hgnn)
        avg_tb = np.mean(mse_tb)
        avg_drift = np.mean(energy_drift)
        improvement = 100 * (avg_tb - avg_hgnn) / avg_tb if avg_tb > 0 else 0
        
        results[f"T{T}"] = {
            "mse_hgnn": avg_hgnn,
            "mse_tb": avg_tb,
            "improvement": improvement,
            "energy_drift": avg_drift,
        }
        
        print(f"{T:>8} {avg_hgnn:>12.2e} {avg_tb:>14.2e} {improvement:>+11.1f}% {avg_drift:>12.4f}")
    
    # Ablation
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
    print(f"T=50: λ=0 MSE={avg_ablation:.2e}, λ>0 MSE={results['T50']['mse_hgnn']:.2e}")
    
    if results['T50']['mse_hgnn'] < avg_ablation:
        print("✓ Interaction term helps")
    else:
        print("✗ Interaction term does not help")
    
    return results


def compare_with_baseline(config: Dict) -> None:
    """
    Compare multi-scale training vs previous trajectory-loss training.
    """
    print("\n" + "=" * 70)
    print("Comparison: Multi-Scale vs Trajectory-Loss Training")
    print("=" * 70)
    
    mu = config.get("mu", 398600.4418)
    dt = config.get("dt", 60.0)
    epsilon = config.get("epsilon", 100.0)
    n_particles = config.get("n_particles", 5)
    
    # Load both models if available
    multiscale_path = Path("hgnn_multiscale.pt")
    baseline_path = Path("hgnn_multibody.pt")
    
    def create_model(phi_scale: float) -> HGNN:
        return HGNN(
            mu=mu,
            interaction_strength=phi_scale,
            interaction_radius=20000.0,
            hidden_dims=[64, 64, 32],
            integrator_type="stormer_verlet",
            dt=dt,
            softplus_output=False,
            normalize_features=True,
        ).float()
    
    print("\nT=100 and T=200 comparison (previous failure points):")
    print("-" * 70)
    
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    twobody_H = TwoBodyHamiltonian(mu=mu)
    integrator = StormerVerlet(dt=dt, create_graph=False)
    
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    for T in [100, 200]:
        mse_tb, mse_multi, mse_base = [], [], []
        
        for seed in range(20):
            q0, p0 = generate_random_ic(n_particles, mu, seed + 2000)
            
            with torch.no_grad():
                # Ground truth
                q_gt, p_gt = q0.clone(), p0.clone()
                for _ in range(T):
                    q_gt, p_gt = integrator.step(q_gt, p_gt, gt_hamiltonian, edge_index)
                
                # Two-body
                q_tb, p_tb = q0.clone(), p0.clone()
                for _ in range(T):
                    q_tb, p_tb = integrator.step(q_tb, p_tb, twobody_H, None)
                
                mse_tb.append(torch.mean((q_tb.float() - q_gt.float())**2).item())
                
                # Multi-scale model
                if multiscale_path.exists():
                    model_multi = create_model(config.get("phi_scale", 0.005))
                    model_multi.load_state_dict(torch.load(multiscale_path, weights_only=True), strict=False)
                    model_multi.eval()
                    q_m, p_m = model_multi(q0.float(), p0.float(), n_steps=T)
                    mse_multi.append(torch.mean((q_m - q_gt.float())**2).item())
                
                # Baseline model
                if baseline_path.exists():
                    model_base = create_model(0.1)
                    model_base.load_state_dict(torch.load(baseline_path, weights_only=True), strict=False)
                    model_base.eval()
                    q_b, p_b = model_base(q0.float(), p0.float(), n_steps=T)
                    mse_base.append(torch.mean((q_b - q_gt.float())**2).item())
        
        avg_tb = np.mean(mse_tb)
        
        print(f"T={T}:")
        print(f"  Two-body MSE: {avg_tb:.2e}")
        
        if mse_multi:
            avg_multi = np.mean(mse_multi)
            imp_multi = 100 * (avg_tb - avg_multi) / avg_tb
            print(f"  Multi-scale MSE: {avg_multi:.2e} ({imp_multi:+.1f}%)")
        
        if mse_base:
            avg_base = np.mean(mse_base)
            imp_base = 100 * (avg_tb - avg_base) / avg_tb
            print(f"  Baseline MSE: {avg_base:.2e} ({imp_base:+.1f}%)")


if __name__ == "__main__":
    config = {
        "mu": 398600.4418,
        "dt": 60.0,
        "n_particles": 5,
        "n_epochs": 150,  # Reduced for faster testing
        "lr": 1e-4,
        "epsilon": 100.0,
        "interaction_radius": 20000.0,
        "batch_size": 5,  # Reduced batch size
        "n_eval": 15,
        # Multi-scale specific
        "horizons": [5, 10, 20, 50],
        "weights": [0.2, 0.3, 0.3, 0.2],
        "lambda_energy": 0.1,
        "phi_scale": 0.005,
    }
    
    model = train_multiscale(config)
    results = evaluate_multiscale(model, config)
    compare_with_baseline(config)
