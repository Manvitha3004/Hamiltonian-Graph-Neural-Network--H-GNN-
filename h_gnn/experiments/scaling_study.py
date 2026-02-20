"""
Object Count Scaling Study.

Tests H-GNN performance with varying number of particles: N ∈ {10, 25, 50, 100}

Measures:
- MSE scaling
- Energy drift scaling  
- Training time scaling
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple
from pathlib import Path

from h_gnn.models import HGNN, TwoBodyHamiltonian
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet


def generate_random_ic(n_particles: int, mu: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random initial conditions."""
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


def train_model_for_scaling(
    n_particles: int,
    config: Dict,
) -> Tuple[HGNN, float]:
    """Train model and return it with training time."""
    mu = config.get("mu", 398600.4418)
    dt = config.get("dt", 60.0)
    epsilon = config.get("epsilon", 100.0)
    n_epochs = config.get("n_epochs", 100)  # Fewer epochs for scaling study
    lr = config.get("lr", 1e-4)
    
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    base_integrator = StormerVerlet(dt=dt, create_graph=False)
    
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Use constant lambda - matched to GT interaction scale
    # GT: epsilon * LJ ~ 0.015 per edge, Model: phi ~ 3
    # So lambda ~ 0.015 / 3 = 0.005
    # Scale training epochs with N to give larger systems more learning time
    n_epochs_scaled = max(n_epochs, n_epochs * n_particles // 10)
    
    model = HGNN(
        mu=mu,
        interaction_strength=0.005,  # Matched to GT interaction scale
        interaction_radius=20000.0,
        hidden_dims=[64, 64, 32],
        integrator_type="stormer_verlet",
        dt=dt,
        create_graph=True,
        softplus_output=False,
        normalize_features=True,
        init_gain=1.0,
    ).float()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train with scaled epochs for larger systems
    start_time = time.time()
    
    for epoch in range(n_epochs_scaled):
        model.train()
        
        q0, p0 = generate_random_ic(n_particles, mu, epoch)
        
        # Forward pass (T=10)
        with torch.no_grad():
            q_true, p_true = q0.clone(), p0.clone()
            for _ in range(10):
                q_true, p_true = base_integrator.step(q_true, p_true, gt_hamiltonian, edge_index)
        
        q_pred, p_pred = model.forward(q0.float(), p0.float(), n_steps=10, return_trajectory=False)
        loss = torch.mean((q_pred - q_true.float())**2 + (p_pred - p_true.float())**2)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
    
    train_time = time.time() - start_time
    
    return model, train_time


def evaluate_scaling(model: HGNN, n_particles: int, config: Dict) -> Dict:
    """Evaluate model at given particle count."""
    mu = config.get("mu", 398600.4418)
    dt = config.get("dt", 60.0)
    epsilon = config.get("epsilon", 100.0)
    n_eval = config.get("n_eval", 10)
    
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    twobody_H = TwoBodyHamiltonian(mu=mu)
    integrator = StormerVerlet(dt=dt, create_graph=False)
    
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    model.eval()
    
    results = {}
    
    for T in [10, 50]:
        mse_hgnn, mse_tb, e_drift = [], [], []
        
        for seed in range(n_eval):
            q0, p0 = generate_random_ic(n_particles, mu, seed + 1000)
            
            with torch.no_grad():
                H0 = gt_hamiltonian(q0, p0).item()
                
                # Ground truth
                q_gt, p_gt = q0.clone(), p0.clone()
                for _ in range(T):
                    q_gt, p_gt = integrator.step(q_gt, p_gt, gt_hamiltonian, edge_index)
                
                # Two-body
                q_tb, p_tb = q0.clone(), p0.clone()
                for _ in range(T):
                    q_tb, p_tb = integrator.step(q_tb, p_tb, twobody_H, None)
                
                # H-GNN
                q_hgnn, p_hgnn = model(q0.float(), p0.float(), n_steps=T)
                
                # Energy at end
                H_end = model.hamiltonian(q_hgnn, p_hgnn, edge_index).item()
            
            mse_hgnn.append(torch.mean((q_hgnn - q_gt.float())**2).item())
            mse_tb.append(torch.mean((q_tb.float() - q_gt.float())**2).item())
            e_drift.append(abs(H_end - H0) / (abs(H0) + 1e-8))
        
        results[f"T{T}_mse_hgnn"] = np.mean(mse_hgnn)
        results[f"T{T}_mse_tb"] = np.mean(mse_tb)
        results[f"T{T}_improvement"] = 100 * (np.mean(mse_tb) - np.mean(mse_hgnn)) / np.mean(mse_tb)
        results[f"T{T}_e_drift"] = np.mean(e_drift)
    
    return results


def run_scaling_study(config: Dict) -> None:
    """Run full scaling study."""
    print("=" * 70)
    print("Object Count Scaling Study")
    print("=" * 70)
    
    particle_counts = config.get("particle_counts", [10, 25, 50])
    
    all_results = []
    
    for n_particles in particle_counts:
        print(f"\n--- N = {n_particles} particles ---")
        n_edges = n_particles * (n_particles - 1)
        print(f"Edges: {n_edges}")
        
        # Train
        print("Training...")
        model, train_time = train_model_for_scaling(n_particles, config)
        print(f"Training time: {train_time:.1f}s")
        
        # Evaluate
        print("Evaluating...")
        results = evaluate_scaling(model, n_particles, config)
        results["n_particles"] = n_particles
        results["n_edges"] = n_edges
        results["train_time"] = train_time
        all_results.append(results)
        
        print(f"T=10: Improvement = {results['T10_improvement']:+.1f}%")
        print(f"T=50: Improvement = {results['T50_improvement']:+.1f}%")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)
    print(f"{'N':>5} {'Edges':>8} {'Time(s)':>10} {'T10 Imp':>10} {'T50 Imp':>10} {'E_drift':>10}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['n_particles']:>5} {r['n_edges']:>8} {r['train_time']:>10.1f} "
              f"{r['T10_improvement']:>+9.1f}% {r['T50_improvement']:>+9.1f}% "
              f"{r['T50_e_drift']:>10.4f}")
    
    # Scaling analysis
    print("\n" + "-" * 70)
    print("Scaling Analysis:")
    
    # Compute scaling exponents
    if len(all_results) >= 2:
        ns = [r["n_particles"] for r in all_results]
        times = [r["train_time"] for r in all_results]
        
        # Log-log fit for training time scaling
        log_n = np.log(ns)
        log_t = np.log(times)
        slope = (log_t[-1] - log_t[0]) / (log_n[-1] - log_n[0])
        
        print(f"Training time scales as O(N^{slope:.2f})")
        if slope < 2:
            print("Sub-quadratic scaling achieved!")
        else:
            print("Quadratic or worse scaling")


if __name__ == "__main__":
    config = {
        "mu": 398600.4418,
        "dt": 60.0,
        "epsilon": 100.0,
        "n_epochs": 100,
        "lr": 1e-4,
        "n_eval": 10,
        "particle_counts": [10, 25, 50],  # Can add 100 if time permits
    }
    
    run_scaling_study(config)
