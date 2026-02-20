"""
Phase-Space Volume Preservation Test for Multi-Body System.

Tests log|det J| over time for:
1. H-GNN (symplectic Störmer-Verlet)
2. Vanilla GNN + RK4 (non-symplectic)

In chaotic multi-body regime, RK4 volume drift will accumulate.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

from h_gnn.models import HGNN, TwoBodyHamiltonian
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet
from h_gnn.experiments.baseline_comparison import AccelerationCorrectionGNN


def compute_log_det_jacobian(
    step_func,
    z: torch.Tensor,
) -> float:
    """
    Compute log|det(J)| for a single step.
    
    For symplectic maps: log|det(J)| = 0
    For non-symplectic: log|det(J)| ≠ 0
    """
    z_input = z.detach().clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(step_func, z_input)
    
    # Handle potential numerical issues
    det = torch.linalg.det(J)
    log_det = torch.log(torch.abs(det) + 1e-30)
    
    return log_det.item()


def run_3body_volume_test(
    epsilon: float = 50.0,
    dt: float = 60.0,
    n_steps: int = 200,
    seed: int = 42,
) -> Dict:
    """
    Compare phase-space volume preservation: H-GNN vs Vanilla GNN.
    
    Uses 3-body gravitational system for exact Jacobian computation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("3-BODY PHASE-SPACE VOLUME TEST")
    print("=" * 70)
    print(f"ε={epsilon}, dt={dt}, n_steps={n_steps}")
    print()
    
    n_particles = 3
    dim = 3
    
    # Create ground truth Hamiltonian
    gt_hamiltonian = MultiBodyHamiltonian(epsilon=epsilon)
    
    # Create H-GNN
    hgnn = HGNN(
        mu=398600.4418,
        interaction_strength=0.1,  # Match training config
        interaction_radius=20000.0,
        hidden_dims=[64, 64, 32],
        integrator_type='stormer_verlet',
        dt=dt,
        softplus_output=False,
        normalize_features=True,
    )
    
    # Load pretrained model if available
    model_path = Path("hgnn_multibody.pt")
    if model_path.exists():
        try:
            hgnn.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
            print("Loaded pretrained H-GNN model")
        except RuntimeError as e:
            print(f"Could not load pretrained model (architecture mismatch), using fresh model")
    
    hgnn.eval()
    
    # Create Vanilla GNN
    vanilla = AccelerationCorrectionGNN(
        mu=398600.4418,
        hidden_dim=64,
        n_layers=2,
        dt=dt,
        interaction_radius=5000.0,
    )
    vanilla.eval()
    
    base_integrator = StormerVerlet(dt=dt, create_graph=False)
    
    # Build edge index
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Initial conditions (3-body)
    np.random.seed(seed)
    r = 7000 + 500 * np.random.randn(n_particles)
    theta = np.linspace(0, 2*np.pi, n_particles, endpoint=False)
    
    q0 = torch.tensor([
        [r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i]), 50.0 * np.random.randn()]
        for i in range(n_particles)
    ], dtype=torch.float32)
    
    v_circ = np.sqrt(398600.4418 / r)
    p0 = torch.tensor([
        [v_circ[i] * (-np.sin(theta[i])), v_circ[i] * np.cos(theta[i]), 0.1 * np.random.randn()]
        for i in range(n_particles)
    ], dtype=torch.float32)
    
    # Flatten to phase-space vector z = [q, p]
    z0 = torch.cat([q0.flatten(), p0.flatten()])
    phase_dim = z0.shape[0]
    
    print(f"Phase space dimension: {phase_dim}")
    print()
    
    # Define step functions
    def hgnn_step(z):
        q = z[:phase_dim//2].reshape(n_particles, dim)
        p = z[phase_dim//2:].reshape(n_particles, dim)
        
        graph = hgnn.graph_builder(q)
        q_new, p_new = hgnn.integrator.step(q, p, hgnn.hamiltonian, graph.edge_index)
        
        return torch.cat([q_new.flatten(), p_new.flatten()])
    
    def vanilla_step(z):
        q = z[:phase_dim//2].reshape(n_particles, dim)
        p = z[phase_dim//2:].reshape(n_particles, dim)
        
        q_new, p_new = vanilla._rk4_step(q, p, edge_index, vanilla.dt)
        
        return torch.cat([q_new.flatten(), p_new.flatten()])
    
    def base_sv_step(z):
        q = z[:phase_dim//2].reshape(n_particles, dim).double()
        p = z[phase_dim//2:].reshape(n_particles, dim).double()
        
        tb = TwoBodyHamiltonian()
        q_new, p_new = base_integrator.step(q, p, tb, None)
        
        return torch.cat([q_new.flatten().float(), p_new.flatten().float()])
    
    # Track cumulative log|det J|
    results = {
        'step': [],
        'log_det_hgnn': [],
        'log_det_vanilla': [],
        'log_det_base': [],
        'cumsum_hgnn': [],
        'cumsum_vanilla': [],
        'cumsum_base': [],
    }
    
    z_hgnn = z0.clone()
    z_vanilla = z0.clone()
    z_base = z0.clone()
    
    cumsum_hgnn = 0.0
    cumsum_vanilla = 0.0
    cumsum_base = 0.0
    
    print("Computing Jacobian determinants per step...")
    print("Step   log|det J| H-GNN   Vanilla   Base SV")
    print("-" * 50)
    
    for step in range(n_steps + 1):
        results['step'].append(step)
        
        if step == 0:
            # Initial state
            log_det_hgnn = 0.0
            log_det_vanilla = 0.0
            log_det_base = 0.0
        else:
            try:
                # Compute log|det J| for current step
                log_det_hgnn = compute_log_det_jacobian(hgnn_step, z_hgnn)
                log_det_vanilla = compute_log_det_jacobian(vanilla_step, z_vanilla)
                log_det_base = compute_log_det_jacobian(base_sv_step, z_base)
            except Exception as e:
                print(f"Step {step} error: {e}")
                log_det_hgnn = 0.0
                log_det_vanilla = 0.0
                log_det_base = 0.0
        
        cumsum_hgnn += log_det_hgnn
        cumsum_vanilla += log_det_vanilla
        cumsum_base += log_det_base
        
        results['log_det_hgnn'].append(log_det_hgnn)
        results['log_det_vanilla'].append(log_det_vanilla)
        results['log_det_base'].append(log_det_base)
        results['cumsum_hgnn'].append(cumsum_hgnn)
        results['cumsum_vanilla'].append(cumsum_vanilla)
        results['cumsum_base'].append(cumsum_base)
        
        if step % 50 == 0:
            print(f"{step:4d}   {log_det_hgnn:+.4f}        {log_det_vanilla:+.4f}    {log_det_base:+.4f}")
        
        # Take step
        if step < n_steps:
            with torch.no_grad():
                z_hgnn = hgnn_step(z_hgnn)
                z_vanilla = vanilla_step(z_vanilla)
                z_base = base_sv_step(z_base)
    
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nCumulative log|det J| after {n_steps} steps:")
    print(f"  H-GNN (Symplectic SV):  {cumsum_hgnn:+.4f}")
    print(f"  Vanilla GNN + RK4:      {cumsum_vanilla:+.4f}")
    print(f"  Base SV (no GNN):       {cumsum_base:+.4f}")
    
    print(f"\nVolume change factor:")
    print(f"  H-GNN:   exp({cumsum_hgnn:.4f}) = {np.exp(cumsum_hgnn):.6f}")
    print(f"  Vanilla: exp({cumsum_vanilla:.4f}) = {np.exp(cumsum_vanilla):.6f}")
    print(f"  Base:    exp({cumsum_base:.4f}) = {np.exp(cumsum_base):.6f}")
    
    print()
    print("INTERPRETATION:")
    print("-" * 50)
    print("log|det J| ≈ 0 → Volume preserving (symplectic)")
    print("log|det J| ≠ 0 → Volume expanding/contracting")
    print()
    if abs(cumsum_hgnn) < abs(cumsum_vanilla):
        print("✓ H-GNN preserves volume better than Vanilla GNN")
    else:
        print("⚠ Vanilla preserves volume better (unexpected)")
    
    # Save results
    output_dir = Path("results/volume_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "volume_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/volume_test_results.json")
    
    return results


if __name__ == "__main__":
    results = run_3body_volume_test(
        epsilon=50.0,
        n_steps=100,
    )
