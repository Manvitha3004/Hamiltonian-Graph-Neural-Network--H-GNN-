"""
Phase-Space Volume Preservation Test.

Symplectic integrators preserve phase-space volume (det(Jacobian) = 1).
Non-symplectic methods (vanilla GNN + RK4) do not.

Uses Hutchinson trace estimator for efficient Jacobian determinant approximation.

If det ≈ 1 for H-GNN and det deviates for vanilla GNN,
this is a deep structural argument for symplectic preservation.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict

from h_gnn.models import HGNN, TwoBodyHamiltonian
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet
from h_gnn.data import SyntheticCatalog
from h_gnn.experiments.baseline_comparison import AccelerationCorrectionGNN, train_vanilla_gnn
from h_gnn.experiments.epsilon_sweep import train_model_for_epsilon


def hutchinson_trace_estimator(
    func,
    x: torch.Tensor,
    n_samples: int = 10,
) -> torch.Tensor:
    """
    Estimate trace of Jacobian using Hutchinson's trace estimator.
    
    For symplectic maps, we expect det(J) = 1, so log(det(J)) = 0.
    Using: log(det(J)) = trace(log(J)) ≈ trace(J - I) for J ≈ I
    
    Actually, for exact det estimation we use:
    det(J) = exp(trace(log(J)))
    
    But for efficiency, we estimate trace(J) which should be ≈ dim for det ≈ 1.
    """
    x = x.detach().requires_grad_(True)
    
    traces = []
    for _ in range(n_samples):
        # Random Rademacher vector
        v = torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2 - 1
        
        # Forward pass
        y = func(x)
        
        # Compute v^T J v using vector-Jacobian product
        vjp, = torch.autograd.grad(y, x, v, create_graph=False)
        trace_est = (v * vjp).sum()
        traces.append(trace_est.item())
    
    return np.mean(traces)


def compute_jacobian_determinant(
    model,
    q: torch.Tensor,
    p: torch.Tensor,
    edge_index: torch.Tensor = None,
    n_particles: int = 5,  # Use subset for computational efficiency
) -> float:
    """
    Compute Jacobian determinant of one-step map.
    
    For a symplectic integrator, det(J) = 1.
    For non-symplectic methods, det(J) ≠ 1.
    
    Uses direct Jacobian computation for small systems.
    """
    # Use subset of particles for efficiency
    q_sub = q[:n_particles].clone().detach()
    p_sub = p[:n_particles].clone().detach()
    
    # Full state vector z = [q, p]
    z = torch.cat([q_sub.flatten(), p_sub.flatten()])
    z.requires_grad_(True)
    
    dim = z.shape[0]
    
    # Define one-step function
    def step_func(z_in):
        q_in = z_in[:dim//2].reshape(-1, 3)
        p_in = z_in[dim//2:].reshape(-1, 3)
        
        if hasattr(model, 'hamiltonian'):
            # H-GNN
            q_out, p_out = model.integrator.step(
                q_in, p_in, model.hamiltonian, edge_index
            )
        else:
            # Vanilla GNN - one RK4 step
            q_out, p_out = model._rk4_step(q_in, p_in, edge_index, model.dt)
        
        return torch.cat([q_out.flatten(), p_out.flatten()])
    
    # Compute Jacobian
    jacobian = torch.zeros(dim, dim, dtype=z.dtype, device=z.device)
    
    for i in range(dim):
        z_grad = z.detach().clone().requires_grad_(True)
        z_out = step_func(z_grad)
        
        grad_mask = torch.zeros(dim, dtype=z.dtype, device=z.device)
        grad_mask[i] = 1.0
        
        grad, = torch.autograd.grad(z_out, z_grad, grad_mask)
        jacobian[:, i] = grad
    
    # Determinant
    det = torch.det(jacobian).item()
    
    return det


def compute_symplectic_deviation(
    model,
    q: torch.Tensor,
    p: torch.Tensor,
    edge_index: torch.Tensor = None,
    n_particles: int = 5,
) -> float:
    """
    Compute how much the map deviates from symplecticity.
    
    For a symplectic map: J^T Ω J = Ω
    where Ω is the symplectic matrix.
    
    Deviation = ||J^T Ω J - Ω||_F
    """
    q_sub = q[:n_particles].clone().detach()
    p_sub = p[:n_particles].clone().detach()
    
    z = torch.cat([q_sub.flatten(), p_sub.flatten()])
    z.requires_grad_(True)
    
    dim = z.shape[0]
    half_dim = dim // 2
    
    def step_func(z_in):
        q_in = z_in[:half_dim].reshape(-1, 3)
        p_in = z_in[half_dim:].reshape(-1, 3)
        
        if hasattr(model, 'hamiltonian'):
            q_out, p_out = model.integrator.step(
                q_in, p_in, model.hamiltonian, edge_index
            )
        else:
            q_out, p_out = model._rk4_step(q_in, p_in, edge_index, model.dt)
        
        return torch.cat([q_out.flatten(), p_out.flatten()])
    
    # Compute Jacobian
    jacobian = torch.zeros(dim, dim, dtype=z.dtype, device=z.device)
    
    for i in range(dim):
        z_grad = z.detach().clone().requires_grad_(True)
        z_out = step_func(z_grad)
        
        grad_mask = torch.zeros(dim, dtype=z.dtype, device=z.device)
        grad_mask[i] = 1.0
        
        grad, = torch.autograd.grad(z_out, z_grad, grad_mask)
        jacobian[:, i] = grad
    
    # Symplectic matrix Ω = [[0, I], [-I, 0]]
    I = torch.eye(half_dim, dtype=z.dtype, device=z.device)
    Z = torch.zeros(half_dim, half_dim, dtype=z.dtype, device=z.device)
    omega = torch.cat([
        torch.cat([Z, I], dim=1),
        torch.cat([-I, Z], dim=1),
    ], dim=0)
    
    # Check J^T Ω J = Ω
    JT_omega_J = jacobian.T @ omega @ jacobian
    deviation = torch.norm(JT_omega_J - omega, p='fro').item()
    
    return deviation


def run_phase_space_analysis(
    epsilon: float = 50.0,
    n_steps_list: List[int] = [1, 5, 10, 25],
    n_seeds: int = 3,
    save_dir: str = "results/phase_space",
) -> Dict:
    """
    Run phase-space volume preservation analysis.
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
    
    print("=" * 70)
    print("PHASE-SPACE VOLUME PRESERVATION ANALYSIS")
    print("=" * 70)
    print(f"ε = {epsilon}")
    print()
    
    results = {
        "hgnn": {"det": [], "symplectic_deviation": []},
        "vanilla": {"det": [], "symplectic_deviation": []},
        "twobody": {"det": [], "symplectic_deviation": []},
    }
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        print("-" * 40)
        
        # Train models
        print("Training H-GNN...")
        hgnn_model, _, _ = train_model_for_epsilon(epsilon, config, seed)
        
        print("Training Vanilla GNN...")
        vanilla_model = train_vanilla_gnn(epsilon, config, seed)
        
        # Test data
        catalog = SyntheticCatalog(mu=mu, seed=seed + 100)
        q_test, p_test = catalog.generate_training_set(10)  # Small for Jacobian
        
        # Build graph
        graph = hgnn_model.graph_builder(q_test)
        edge_index = graph.edge_index
        
        # Also need a two-body only baseline
        tb_H = TwoBodyHamiltonian(mu=mu)
        
        print("\nComputing Jacobian determinants (5 particles)...")
        
        # H-GNN
        det_hgnn = compute_jacobian_determinant(
            hgnn_model, q_test, p_test, edge_index, n_particles=5
        )
        dev_hgnn = compute_symplectic_deviation(
            hgnn_model, q_test, p_test, edge_index, n_particles=5
        )
        results["hgnn"]["det"].append(det_hgnn)
        results["hgnn"]["symplectic_deviation"].append(dev_hgnn)
        
        # Vanilla GNN
        det_vanilla = compute_jacobian_determinant(
            vanilla_model, q_test, p_test, edge_index, n_particles=5
        )
        dev_vanilla = compute_symplectic_deviation(
            vanilla_model, q_test, p_test, edge_index, n_particles=5
        )
        results["vanilla"]["det"].append(det_vanilla)
        results["vanilla"]["symplectic_deviation"].append(dev_vanilla)
        
        print(f"  H-GNN:   det = {det_hgnn:.6f}, symplectic deviation = {dev_hgnn:.6f}")
        print(f"  Vanilla: det = {det_vanilla:.6f}, symplectic deviation = {dev_vanilla:.6f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE-SPACE PRESERVATION SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':>12} {'Mean det':>12} {'Std det':>10} {'|det-1|':>10} {'Symp Dev':>12}")
    print("-" * 60)
    
    for model_name in ["hgnn", "vanilla"]:
        dets = results[model_name]["det"]
        devs = results[model_name]["symplectic_deviation"]
        
        mean_det = np.mean(dets)
        std_det = np.std(dets)
        det_error = abs(mean_det - 1.0)
        mean_dev = np.mean(devs)
        
        print(f"{model_name:>12} {mean_det:>12.6f} {std_det:>10.6f} {det_error:>10.6f} {mean_dev:>12.6f}")
    
    # Interpretation
    print("\nInterpretation:")
    hgnn_det_error = abs(np.mean(results["hgnn"]["det"]) - 1.0)
    vanilla_det_error = abs(np.mean(results["vanilla"]["det"]) - 1.0)
    
    if hgnn_det_error < vanilla_det_error:
        ratio = vanilla_det_error / hgnn_det_error if hgnn_det_error > 0 else float('inf')
        print(f"  H-GNN preserves phase-space volume {ratio:.1f}x better than Vanilla GNN")
    
    hgnn_dev = np.mean(results["hgnn"]["symplectic_deviation"])
    vanilla_dev = np.mean(results["vanilla"]["symplectic_deviation"])
    
    if hgnn_dev < vanilla_dev:
        ratio = vanilla_dev / hgnn_dev if hgnn_dev > 0 else float('inf')
        print(f"  H-GNN symplectic deviation {ratio:.1f}x smaller than Vanilla GNN")
    
    # Save
    results_file = Path(save_dir) / "phase_space_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = run_phase_space_analysis(
        epsilon=50.0,
        n_seeds=3,
    )
