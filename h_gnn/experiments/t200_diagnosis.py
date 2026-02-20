"""
T=200 Dip Diagnosis.

Hypothesis: Correction magnitude has a slight bias leading to orbital 
precession drift that peaks at ~200 steps before diffusing at 500.

Diagnostics:
1. Per-step local error growth curve
2. Correction magnitude vs step index
3. Phase error (angle drift) accumulation
4. Energy metrics: H_θ drift vs H_true drift
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from h_gnn.models import HGNN, TwoBodyHamiltonian
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet


def compute_phase_angle(q: torch.Tensor) -> torch.Tensor:
    """Compute orbital phase angle for each particle.
    
    For 2D orbit, angle = atan2(y, x)
    For 3D orbit, angle in orbital plane = atan2(y, x)
    """
    return torch.atan2(q[:, 1], q[:, 0])


def compute_angular_momentum(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Compute angular momentum magnitude for each particle."""
    # L = r × p
    L = torch.cross(q, p, dim=-1)
    return torch.norm(L, dim=-1)


def diagnose_t200_dip(
    epsilon: float = 50.0,
    n_particles: int = 5,
    dt: float = 60.0,
    max_steps: int = 500,
    n_trajectories: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Comprehensive diagnosis of the T=200 performance dip.
    
    Returns detailed metrics at each time step.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("T=200 DIP DIAGNOSIS")
    print("=" * 70)
    print(f"ε={epsilon}, N={n_particles}, dt={dt}, max_steps={max_steps}")
    print()
    
    # Create ground truth Hamiltonian
    gt_hamiltonian = MultiBodyHamiltonian(epsilon=epsilon)
    tb_hamiltonian = TwoBodyHamiltonian()
    
    # Create H-GNN (pre-trained)
    hgnn = HGNN(
        mu=398600.4418,
        interaction_strength=1.0,
        interaction_radius=5000.0,
        hidden_dims=[128, 128, 64],
        integrator_type='stormer_verlet',
        dt=dt,
        softplus_output=False,
        normalize_features=True,
        create_graph=True,
    ).double()
    
    # Load pretrained model if available
    model_path = Path("hgnn_multibody.pt")
    if model_path.exists():
        hgnn.load_state_dict(torch.load(model_path, weights_only=True))
        print("Loaded pretrained model from hgnn_multibody.pt")
    else:
        print("WARNING: No pretrained model found. Using random weights.")
    
    hgnn.eval()
    
    base_integrator = StormerVerlet(dt=dt, create_graph=False)
    
    # Build edge index
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Store metrics per step
    all_metrics = {
        'step': list(range(max_steps + 1)),
        'mse_twobody': [],  # MSE of two-body vs true
        'mse_hgnn': [],     # MSE of H-GNN vs true
        'improvement': [],  # % improvement of H-GNN over two-body
        'phase_error_twobody': [],    # Phase angle error (radians)
        'phase_error_hgnn': [],
        'correction_magnitude': [],   # ||Φ contribution||
        'base_gradient_magnitude': [],
        'correction_ratio': [],       # ||∇Φ|| / ||∇H_base||
        'energy_drift_learned': [],   # H_θ(t) - H_θ(0)
        'energy_drift_true': [],      # H_true(t) - H_true(0)
        'energy_drift_twobody': [],   # For two-body baseline
        'angular_momentum_drift_hgnn': [],
        'angular_momentum_drift_twobody': [],
    }
    
    for traj_idx in range(n_trajectories):
        print(f"\nTrajectory {traj_idx + 1}/{n_trajectories}")
        
        # Random initial conditions
        np.random.seed(seed + traj_idx)
        r = 7000 + 500 * np.random.randn(n_particles)
        theta = np.linspace(0, 2*np.pi, n_particles, endpoint=False) + 0.1 * np.random.randn(n_particles)
        
        q0 = np.array([
            [r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i]), 50.0 * np.random.randn()]
            for i in range(n_particles)
        ])
        v_circ = np.sqrt(398600.4418 / r)
        p0 = np.array([
            [v_circ[i] * (-np.sin(theta[i])), v_circ[i] * np.cos(theta[i]), 0.1 * np.random.randn()]
            for i in range(n_particles)
        ])
        
        q0_tensor = torch.tensor(q0, dtype=torch.float64)
        p0_tensor = torch.tensor(p0, dtype=torch.float64)
        
        # Initialize trajectories
        q_true, p_true = q0_tensor.clone(), p0_tensor.clone()
        q_tb, p_tb = q0_tensor.clone(), p0_tensor.clone()
        q_hgnn, p_hgnn = q0_tensor.clone().float(), p0_tensor.clone().float()
        
        # Initial energies
        with torch.no_grad():
            H_true_0 = gt_hamiltonian(q_true, p_true, sum_over_particles=True).item()
            H_theta_0 = hgnn.hamiltonian(q_hgnn, p_hgnn, edge_index).item()
            L_0_hgnn = compute_angular_momentum(q_hgnn, p_hgnn).mean().item()
            L_0_tb = compute_angular_momentum(q_tb.float(), p_tb.float()).mean().item()
        
        # Initial phase angles
        phase_0 = compute_phase_angle(q0_tensor.float())
        
        # Step-by-step integration
        step_metrics = []
        
        for step in range(max_steps + 1):
            if step == 0:
                # Step 0 is initial condition
                mse_tb = 0.0
                mse_hgnn = 0.0
                phase_err_tb = 0.0
                phase_err_hgnn = 0.0
            else:
                # Compute MSE
                mse_tb = torch.mean((q_tb.float() - q_true.float())**2).item()
                mse_hgnn = torch.mean((q_hgnn - q_true.float())**2).item()
                
                # Compute phase error (mean absolute angle difference)
                phase_true = compute_phase_angle(q_true.float())
                phase_tb = compute_phase_angle(q_tb.float())
                phase_hgnn = compute_phase_angle(q_hgnn)
                
                # Angle wrapping: compute minimum angular distance
                def angle_diff(a, b):
                    diff = torch.abs(a - b)
                    diff = torch.min(diff, 2*np.pi - diff)
                    return diff.mean().item()
                
                phase_err_tb = angle_diff(phase_tb, phase_true)
                phase_err_hgnn = angle_diff(phase_hgnn, phase_true)
            
            # Compute correction magnitude (at current state)
            with torch.no_grad():
                # Get gradients from full Hamiltonian
                grad_q_full, _ = hgnn.hamiltonian.compute_gradients(
                    q_hgnn, p_hgnn, edge_index, create_graph=False
                )
                # Get gradients from base Hamiltonian
                grad_q_base, _ = tb_hamiltonian.compute_gradients(
                    q_hgnn.double(), p_hgnn.double()
                )
                grad_q_base = grad_q_base.float()
                
                # Correction = full - base
                correction = grad_q_full - grad_q_base
                corr_mag = torch.norm(correction).item()
                base_mag = torch.norm(grad_q_base).item()
                
                # Energies
                H_true = gt_hamiltonian(q_true, p_true, sum_over_particles=True).item()
                H_theta = hgnn.hamiltonian(q_hgnn, p_hgnn, edge_index).item()
                H_tb = tb_hamiltonian(q_tb.double(), p_tb.double(), sum_over_particles=True).item()
                
                # Angular momentum
                L_hgnn = compute_angular_momentum(q_hgnn, p_hgnn).mean().item()
                L_tb = compute_angular_momentum(q_tb.float(), p_tb.float()).mean().item()
            
            step_metrics.append({
                'step': step,
                'mse_twobody': mse_tb,
                'mse_hgnn': mse_hgnn,
                'improvement': 100 * (mse_tb - mse_hgnn) / (mse_tb + 1e-10) if mse_tb > 0 else 0,
                'phase_error_twobody': phase_err_tb,
                'phase_error_hgnn': phase_err_hgnn,
                'correction_magnitude': corr_mag,
                'base_gradient_magnitude': base_mag,
                'correction_ratio': corr_mag / (base_mag + 1e-10),
                'energy_drift_learned': H_theta - H_theta_0,
                'energy_drift_true': H_true - H_true_0,
                'energy_drift_twobody': H_tb - H_true_0,
                'angular_momentum_drift_hgnn': (L_hgnn - L_0_hgnn) / (L_0_hgnn + 1e-10),
                'angular_momentum_drift_twobody': (L_tb - L_0_tb) / (L_0_tb + 1e-10),
            })
            
            # Integrate one step
            if step < max_steps:
                with torch.no_grad():
                    q_true, p_true = base_integrator.step(q_true, p_true, gt_hamiltonian, edge_index)
                    q_tb, p_tb = base_integrator.step(q_tb, p_tb, tb_hamiltonian, None)
                
                # H-GNN step (need to rebuild graph since positions changed)
                with torch.no_grad():
                    graph = hgnn.graph_builder(q_hgnn)
                    q_hgnn, p_hgnn = hgnn.integrator.step(
                        q_hgnn, p_hgnn, hgnn.hamiltonian, graph.edge_index
                    )
            
            # Progress
            if step % 100 == 0:
                print(f"  Step {step}: MSE_tb={mse_tb:.2e}, MSE_hgnn={mse_hgnn:.2e}, phase_err={phase_err_hgnn:.4f}")
        
        # Aggregate metrics
        if traj_idx == 0:
            for key in all_metrics:
                if key != 'step':
                    all_metrics[key] = [m[key] for m in step_metrics]
        else:
            for key in all_metrics:
                if key != 'step':
                    for i, m in enumerate(step_metrics):
                        all_metrics[key][i] += m[key]
    
    # Average over trajectories
    for key in all_metrics:
        if key != 'step':
            all_metrics[key] = [x / n_trajectories for x in all_metrics[key]]
    
    # Identify T=200 dip characteristics
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    
    # Find peak error region
    imp = all_metrics['improvement']
    
    # Find local minimum around T=200
    imp_100 = imp[100] if len(imp) > 100 else 0
    imp_200 = imp[200] if len(imp) > 200 else 0
    imp_300 = imp[300] if len(imp) > 300 else 0
    imp_500 = imp[500] if len(imp) > 500 else imp[-1]
    
    print(f"\nImprovement at key horizons:")
    print(f"  T=100: {imp_100:+.1f}%")
    print(f"  T=200: {imp_200:+.1f}%")
    print(f"  T=300: {imp_300:+.1f}%")
    print(f"  T=500: {imp_500:+.1f}%")
    
    # Phase error analysis
    phase_100 = all_metrics['phase_error_hgnn'][100] if len(all_metrics['phase_error_hgnn']) > 100 else 0
    phase_200 = all_metrics['phase_error_hgnn'][200] if len(all_metrics['phase_error_hgnn']) > 200 else 0
    phase_300 = all_metrics['phase_error_hgnn'][300] if len(all_metrics['phase_error_hgnn']) > 300 else 0
    
    print(f"\nPhase error (H-GNN) at key horizons:")
    print(f"  T=100: {phase_100:.4f} rad ({np.degrees(phase_100):.2f}°)")
    print(f"  T=200: {phase_200:.4f} rad ({np.degrees(phase_200):.2f}°)")
    print(f"  T=300: {phase_300:.4f} rad ({np.degrees(phase_300):.2f}°)")
    
    # Correction ratio analysis
    corr_ratio = all_metrics['correction_ratio']
    corr_mean = np.mean(corr_ratio[10:100])
    corr_std = np.std(corr_ratio[10:100])
    
    print(f"\nCorrection ratio ||∇Φ||/||∇H_base||:")
    print(f"  Mean (T=10-100): {corr_mean:.4f} ± {corr_std:.4f}")
    print(f"  T=200: {corr_ratio[200] if len(corr_ratio) > 200 else 0:.4f}")
    
    # Energy drift comparison
    e_drift_learned = all_metrics['energy_drift_learned']
    e_drift_true = all_metrics['energy_drift_true']
    
    print(f"\nEnergy drift at T=200:")
    print(f"  H_θ drift:    {e_drift_learned[200] if len(e_drift_learned) > 200 else 0:.2e}")
    print(f"  H_true drift: {e_drift_true[200] if len(e_drift_true) > 200 else 0:.2e}")
    
    # Hypothesis testing
    print("\n" + "-" * 70)
    print("HYPOTHESIS EVALUATION")
    print("-" * 70)
    
    # Check if phase error correlates with improvement dip
    phase_err = all_metrics['phase_error_hgnn']
    phase_err_tb = all_metrics['phase_error_twobody']
    
    # Relative phase error advantage
    phase_advantage = []
    for i in range(len(phase_err)):
        if phase_err_tb[i] > 0:
            phase_advantage.append(100 * (phase_err_tb[i] - phase_err[i]) / phase_err_tb[i])
        else:
            phase_advantage.append(0)
    
    print(f"\nPhase error advantage (H-GNN vs TwoBody):")
    print(f"  T=100: {phase_advantage[100] if len(phase_advantage) > 100 else 0:+.1f}%")
    print(f"  T=200: {phase_advantage[200] if len(phase_advantage) > 200 else 0:+.1f}%")
    print(f"  T=500: {phase_advantage[500] if len(phase_advantage) > 500 else phase_advantage[-1]:+.1f}%")
    
    # Save results
    output_dir = Path("results/t200_diagnosis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "diagnosis_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/diagnosis_results.json")
    
    # Generate plots
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        steps = all_metrics['step']
        
        # Plot 1: Improvement over time
        ax = axes[0, 0]
        ax.plot(steps, all_metrics['improvement'], 'b-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=200, color='r', linestyle='--', alpha=0.5, label='T=200')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('H-GNN Improvement over Two-Body')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Phase error
        ax = axes[0, 1]
        ax.plot(steps, np.degrees(all_metrics['phase_error_hgnn']), 'b-', label='H-GNN')
        ax.plot(steps, np.degrees(all_metrics['phase_error_twobody']), 'r--', label='Two-Body')
        ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Phase Error (degrees)')
        ax.set_title('Orbital Phase Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Correction ratio
        ax = axes[0, 2]
        ax.plot(steps, all_metrics['correction_ratio'], 'g-', linewidth=1.5)
        ax.axvline(x=200, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('||∇Φ|| / ||∇H_base||')
        ax.set_title('Correction Magnitude Ratio')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: MSE comparison
        ax = axes[1, 0]
        ax.semilogy(steps[1:], all_metrics['mse_twobody'][1:], 'r-', label='Two-Body')
        ax.semilogy(steps[1:], all_metrics['mse_hgnn'][1:], 'b-', label='H-GNN')
        ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('MSE (km²)')
        ax.set_title('Position Error (MSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Energy drift
        ax = axes[1, 1]
        ax.plot(steps, all_metrics['energy_drift_learned'], 'b-', label='H_θ drift')
        ax.plot(steps, all_metrics['energy_drift_true'], 'r--', label='H_true drift')
        ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Energy Drift')
        ax.set_title('Energy Conservation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Angular momentum drift
        ax = axes[1, 2]
        ax.plot(steps, all_metrics['angular_momentum_drift_hgnn'], 'b-', label='H-GNN')
        ax.plot(steps, all_metrics['angular_momentum_drift_twobody'], 'r--', label='Two-Body')
        ax.axvline(x=200, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Relative L drift')
        ax.set_title('Angular Momentum Conservation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "diagnosis_plots.png", dpi=150)
        print(f"Plots saved to {output_dir}/diagnosis_plots.png")
        plt.close()
        
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    return all_metrics


if __name__ == "__main__":
    results = diagnose_t200_dip(
        epsilon=50.0,
        n_particles=5,
        max_steps=500,
        n_trajectories=3,
    )
