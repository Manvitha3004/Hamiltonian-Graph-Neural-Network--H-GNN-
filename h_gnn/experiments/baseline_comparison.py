"""
Baseline Comparison Experiment.

Compares H-GNN (symplectic) against:
1. Pure two-body physics
2. Vanilla GNN + RK4 (non-symplectic)

Key hypothesis: Symplectic structure provides long-horizon stability.

Architecture comparisons:
- H-GNN: H = H_twobody + λΦ, integrated with Störmer-Verlet
- Vanilla GNN: a = a_twobody + MPNN(q,p), integrated with RK4
- Two-body: Pure Kepler physics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

from h_gnn.models import HGNN, TwoBodyHamiltonian, GraphBuilder
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet
from h_gnn.integrators.baselines import RK4
from h_gnn.data import SyntheticCatalog


class AccelerationCorrectionGNN(nn.Module):
    """
    Non-symplectic GNN that predicts acceleration corrections.
    
    Architecture:
        a_i = a_twobody(q) + MPNN(q, p)
    
    Integrated with RK4, NOT symplectic integrator.
    This is a fair baseline that also uses two-body physics as foundation.
    """
    
    def __init__(
        self,
        mu: float = 398600.4418,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dt: float = 60.0,
        interaction_radius: float = 5000.0,
    ):
        super().__init__()
        
        self.mu = mu
        self.dt = dt
        self.hidden_dim = hidden_dim
        
        # Two-body acceleration
        self.tb_H = TwoBodyHamiltonian(mu=mu)
        
        # Graph builder
        self.graph_builder = GraphBuilder(
            radius=interaction_radius,
            max_neighbors=50,
        )
        
        # MPNN for acceleration correction
        # Input: invariant edge features
        self.edge_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.message_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.update_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        
        # Output: acceleration correction (per node)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # 3D acceleration
        )
        
        # Initialize for small initial corrections
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def compute_acceleration(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute total acceleration: a = a_twobody + a_correction.
        """
        N = q.shape[0]
        
        # Two-body acceleration: a = -∂H/∂q = μq/|q|³
        r = torch.norm(q, dim=-1, keepdim=True) + 1e-8
        a_twobody = -self.mu * q / (r ** 3)
        
        if edge_index is None or edge_index.shape[1] == 0:
            return a_twobody
        
        src, dst = edge_index
        
        # Compute edge features (invariant)
        delta_q = q[dst] - q[src]
        delta_p = p[dst] - p[src]
        p_avg = 0.5 * (p[src] + p[dst])
        
        r_pair = torch.norm(delta_q, dim=-1, keepdim=True) + 1e-8
        r_sq = r_pair ** 2
        delta_p_norm = torch.norm(delta_p, dim=-1, keepdim=True)
        dot_qp = (delta_q * delta_p).sum(dim=-1, keepdim=True)
        dot_q_pavg = (delta_q * p_avg).sum(dim=-1, keepdim=True)
        
        # Normalize features
        edge_feat_raw = torch.cat([
            r_pair / 1000.0,
            r_sq / 1e6,
            delta_p_norm / 10.0,
            dot_qp / 1e4,
            dot_q_pavg / 1e4,
        ], dim=-1)
        
        edge_feat = self.edge_encoder(edge_feat_raw)  # (E, hidden)
        
        # Initial node hidden state (zeros)
        h = torch.zeros(N, self.hidden_dim, device=q.device, dtype=q.dtype)
        
        # Message passing
        for msg_layer, upd_layer in zip(self.message_layers, self.update_layers):
            # Messages
            msg_input = torch.cat([h[src], h[dst], edge_feat], dim=-1)
            messages = msg_layer(msg_input)
            
            # Aggregate
            agg = torch.zeros(N, self.hidden_dim, device=q.device, dtype=q.dtype)
            agg.index_add_(0, dst, messages)
            
            # Update
            h = upd_layer(torch.cat([h, agg], dim=-1))
        
        # Decode to acceleration correction
        a_correction = self.decoder(h)
        
        return a_twobody + a_correction
    
    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n_steps: int = 1,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate using RK4 with learned acceleration correction.
        """
        # Build graph
        graph = self.graph_builder(q)
        edge_index = graph.edge_index
        
        if return_trajectory:
            q_traj = [q.clone()]
            p_traj = [p.clone()]
        
        q_curr, p_curr = q.clone(), p.clone()
        dt = self.dt
        
        for _ in range(n_steps):
            # RK4 step with neural acceleration
            q_curr, p_curr = self._rk4_step(q_curr, p_curr, edge_index, dt)
            
            if return_trajectory:
                q_traj.append(q_curr.clone())
                p_traj.append(p_curr.clone())
        
        if return_trajectory:
            return torch.stack(q_traj), torch.stack(p_traj)
        return q_curr, p_curr
    
    def _rk4_step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single RK4 integration step."""
        
        # k1
        a1 = self.compute_acceleration(q, p, edge_index)
        dq1 = p
        dp1 = a1
        
        # k2
        q2 = q + 0.5 * dt * dq1
        p2 = p + 0.5 * dt * dp1
        a2 = self.compute_acceleration(q2, p2, edge_index)
        dq2 = p2
        dp2 = a2
        
        # k3
        q3 = q + 0.5 * dt * dq2
        p3 = p + 0.5 * dt * dp2
        a3 = self.compute_acceleration(q3, p3, edge_index)
        dq3 = p3
        dp3 = a3
        
        # k4
        q4 = q + dt * dq3
        p4 = p + dt * dp3
        a4 = self.compute_acceleration(q4, p4, edge_index)
        dq4 = p4
        dp4 = a4
        
        # Weighted average
        q_new = q + (dt / 6.0) * (dq1 + 2*dq2 + 2*dq3 + dq4)
        p_new = p + (dt / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
        
        return q_new, p_new


def train_vanilla_gnn(
    epsilon: float,
    config: Dict,
    seed: int = 42,
) -> AccelerationCorrectionGNN:
    """
    Train vanilla GNN to match multi-body trajectories.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    mu = config["mu"]
    dt = config["dt"]
    n_objects = config["n_objects"]
    n_epochs = config["n_epochs"]
    lr = config["lr"]
    interaction_radius = config["interaction_radius"]
    rollout_steps = config.get("rollout_steps", 5)
    
    # Generate training data
    catalog = SyntheticCatalog(mu=mu, seed=seed)
    q_train, p_train = catalog.generate_training_set(n_objects)
    
    # Ground truth
    gt_H = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    gt_integrator = StormerVerlet(dt=dt)
    
    # Model
    model = AccelerationCorrectionGNN(
        mu=mu,
        hidden_dim=128,
        n_layers=3,
        dt=dt,
        interaction_radius=interaction_radius,
    ).double()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        
        # Ground truth trajectory
        with torch.no_grad():
            q_gt, p_gt = gt_integrator.integrate(
                q_train, p_train, gt_H, rollout_steps, store_trajectory=True
            )
        
        # Model prediction
        q_pred, p_pred = model(q_train.clone(), p_train.clone(), 
                               n_steps=rollout_steps, return_trajectory=True)
        
        # Trajectory loss
        loss = torch.mean((q_pred - q_gt) ** 2) + torch.mean((p_pred - p_gt) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Loss={loss.item():.6f}")
    
    model.load_state_dict(best_state)
    model.eval()
    
    return model


@dataclass
class BaselineComparisonResult:
    """Results for baseline comparison."""
    horizon: int
    
    # MSE vs multi-body ground truth
    mse_hgnn: float
    mse_vanilla_gnn: float
    mse_twobody: float
    
    # Improvement over two-body
    improvement_hgnn: float
    improvement_vanilla: float
    
    # Energy drift (absolute)
    energy_drift_hgnn: float
    energy_drift_vanilla: float
    energy_drift_twobody: float


def run_baseline_comparison(
    epsilon: float = 50.0,
    horizons: List[int] = [10, 25, 50, 100, 200],
    n_seeds: int = 3,
    save_dir: str = "results/baseline_comparison",
) -> List[BaselineComparisonResult]:
    """
    Run comprehensive baseline comparison.
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
    
    print("=" * 70)
    print("BASELINE COMPARISON: H-GNN vs Vanilla GNN vs Two-Body")
    print("=" * 70)
    print(f"ε = {epsilon}, Horizons = {horizons}, Seeds = {n_seeds}")
    print()
    
    all_results = []
    
    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed + 1}/{n_seeds}")
        print("=" * 70)
        
        # Train H-GNN
        print("\nTraining H-GNN (symplectic)...")
        from h_gnn.experiments.epsilon_sweep import train_model_for_epsilon
        hgnn_model, _, _ = train_model_for_epsilon(epsilon, config, seed)
        
        # Train Vanilla GNN
        print("\nTraining Vanilla GNN (non-symplectic)...")
        vanilla_model = train_vanilla_gnn(epsilon, config, seed)
        
        # Evaluate at multiple horizons
        mu = config["mu"]
        dt = config["dt"]
        
        gt_H = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
        tb_H = TwoBodyHamiltonian(mu=mu)
        sv_integrator = StormerVerlet(dt=dt)
        
        # Test data
        catalog = SyntheticCatalog(mu=mu, seed=seed + 100)
        q_test, p_test = catalog.generate_training_set(100)
        
        print("\nEvaluating...")
        
        for horizon in horizons:
            with torch.no_grad():
                # Ground truth (multi-body)
                q_gt, p_gt = sv_integrator.integrate(
                    q_test, p_test, gt_H, horizon, store_trajectory=True
                )
                
                # H-GNN
                q_hgnn, p_hgnn = hgnn_model(
                    q_test.clone(), p_test.clone(), n_steps=horizon, return_trajectory=True
                )
                
                # Vanilla GNN
                q_vanilla, p_vanilla = vanilla_model(
                    q_test.clone(), p_test.clone(), n_steps=horizon, return_trajectory=True
                )
                
                # Two-body
                q_tb, p_tb = sv_integrator.integrate(
                    q_test, p_test, tb_H, horizon, store_trajectory=True
                )
            
            # MSE
            mse_hgnn = torch.mean((q_hgnn[-1] - q_gt[-1]) ** 2).item()
            mse_vanilla = torch.mean((q_vanilla[-1] - q_gt[-1]) ** 2).item()
            mse_tb = torch.mean((q_tb[-1] - q_gt[-1]) ** 2).item()
            
            # Improvement
            imp_hgnn = (1 - mse_hgnn / mse_tb) * 100 if mse_tb > 0 else 0
            imp_vanilla = (1 - mse_vanilla / mse_tb) * 100 if mse_tb > 0 else 0
            
            # Energy drift (using ground truth Hamiltonian)
            E_init = gt_H(q_gt[0], p_gt[0]).item()
            E_hgnn = gt_H(q_hgnn[-1], p_hgnn[-1]).item()
            E_vanilla = gt_H(q_vanilla[-1], p_vanilla[-1]).item()
            E_tb = gt_H(q_tb[-1], p_tb[-1]).item()
            
            drift_hgnn = abs(E_hgnn - E_init)
            drift_vanilla = abs(E_vanilla - E_init)
            drift_tb = abs(E_tb - E_init)
            
            result = BaselineComparisonResult(
                horizon=horizon,
                mse_hgnn=mse_hgnn,
                mse_vanilla_gnn=mse_vanilla,
                mse_twobody=mse_tb,
                improvement_hgnn=imp_hgnn,
                improvement_vanilla=imp_vanilla,
                energy_drift_hgnn=drift_hgnn,
                energy_drift_vanilla=drift_vanilla,
                energy_drift_twobody=drift_tb,
            )
            
            all_results.append(result)
            
            print(f"  T={horizon:3d}: H-GNN={imp_hgnn:+6.1f}%, "
                  f"Vanilla={imp_vanilla:+6.1f}%, "
                  f"E_drift: H-GNN={drift_hgnn:.2e}, Vanilla={drift_vanilla:.2e}")
    
    # Aggregate results by horizon
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean over seeds)")
    print("=" * 70)
    print(f"{'Horizon':>8} {'H-GNN Imp':>12} {'Vanilla Imp':>12} "
          f"{'H-GNN E_drift':>14} {'Vanilla E_drift':>16}")
    print("-" * 70)
    
    for horizon in horizons:
        horizon_results = [r for r in all_results if r.horizon == horizon]
        
        mean_imp_hgnn = np.mean([r.improvement_hgnn for r in horizon_results])
        mean_imp_vanilla = np.mean([r.improvement_vanilla for r in horizon_results])
        mean_drift_hgnn = np.mean([r.energy_drift_hgnn for r in horizon_results])
        mean_drift_vanilla = np.mean([r.energy_drift_vanilla for r in horizon_results])
        
        print(f"{horizon:>8} {mean_imp_hgnn:>+11.1f}% {mean_imp_vanilla:>+11.1f}% "
              f"{mean_drift_hgnn:>14.2e} {mean_drift_vanilla:>16.2e}")
    
    # Save results
    results_file = Path(save_dir) / "comparison_results.json"
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    results = run_baseline_comparison(
        epsilon=50.0,
        horizons=[10, 25, 50, 100, 200],
        n_seeds=3,
    )
