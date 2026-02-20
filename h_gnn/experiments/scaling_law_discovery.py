"""
Scaling Law Discovery Framework for H-GNN.

Goal: Discover fundamental scaling laws and structural limitations in 
Hamiltonian correction learning.

Implements:
1. Learnability threshold analysis (epsilon sweep)
2. System size scaling (N sweep)
3. Chaos and Lyapunov analysis
4. Backward error dt validation
5. Failure mode characterization

This is NOT about performance optimization.
This is about scientific discovery of theoretical limits.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time
from collections import defaultdict
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from h_gnn.models.hgnn import HGNN
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators.symplectic import StormerVerlet


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================

@dataclass
class TrainingMetrics:
    """Metrics tracked during a single training run."""
    seed: int
    epsilon: float
    n_particles: int
    dt: float
    horizon: int
    
    # Core metrics
    final_improvement: float = 0.0
    final_alpha: float = 0.0
    final_loss: float = float('inf')
    
    # Gradient metrics (per epoch)
    grad_phi_norms: List[float] = field(default_factory=list)
    grad_hbase_norms: List[float] = field(default_factory=list)
    gradient_snr: List[float] = field(default_factory=list)
    
    # Convergence metrics
    loss_history: List[float] = field(default_factory=list)
    alpha_history: List[float] = field(default_factory=list)
    convergence_epoch: int = -1  # Epoch where loss stabilized
    
    # Energy metrics
    energy_drift_history: List[float] = field(default_factory=list)
    
    # Failure indicators
    phi_collapsed: bool = False
    alpha_diverged: bool = False
    training_failed: bool = False
    failure_reason: str = ""


@dataclass
class ScalingLawResult:
    """Results from scaling law analysis."""
    parameter_name: str  # 'epsilon', 'N', 'dt', 'T'
    parameter_values: List[float] = field(default_factory=list)
    mean_improvements: List[float] = field(default_factory=list)
    std_improvements: List[float] = field(default_factory=list)
    mean_alphas: List[float] = field(default_factory=list)
    std_alphas: List[float] = field(default_factory=list)
    
    # Fitted scaling law
    scaling_exponent: float = 0.0
    scaling_coefficient: float = 0.0
    critical_value: float = 0.0  # epsilon_crit, N_crit, etc.
    r_squared: float = 0.0


# ===========================================================================
# CORE TRAINING WITH LOGGING
# ===========================================================================

def generate_ic(n_particles: int, mu: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate initial conditions for n particles."""
    rng = np.random.default_rng(seed)
    q_list, p_list = [], []
    
    for _ in range(n_particles):
        r = rng.uniform(7000, 42000)
        theta = rng.uniform(0, 2 * np.pi)
        q = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
        v_circ = np.sqrt(mu / r)
        v_perturb = rng.uniform(-0.05, 0.05)
        v = (v_circ + v_perturb) * np.array([-np.sin(theta), np.cos(theta), 0.0])
        q_list.append(q)
        p_list.append(v)
    
    return (torch.tensor(np.stack(q_list), dtype=torch.float64),
            torch.tensor(np.stack(p_list), dtype=torch.float64))


def train_with_logging(
    config: Dict,
    verbose: bool = False,
) -> Tuple[HGNN, TrainingMetrics]:
    """
    Train H-GNN with comprehensive gradient and metric logging.
    
    Returns:
        model: Trained model
        metrics: Complete training metrics for scaling law analysis
    """
    # Extract config
    seed = config.get('seed', 42)
    epsilon = config.get('epsilon', 100.0)
    n_particles = config.get('n_particles', 10)
    dt = config.get('dt', 60.0)
    horizon = config.get('horizon', 50)
    mu = config.get('mu', 398600.4418)
    n_epochs = config.get('n_epochs', 150)
    lr = config.get('lr', 5e-5)
    batch_size = config.get('batch_size', 8)
    interaction_radius = config.get('interaction_radius', 20000.0)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize metrics
    metrics = TrainingMetrics(
        seed=seed,
        epsilon=epsilon,
        n_particles=n_particles,
        dt=dt,
        horizon=horizon,
    )
    
    # Ground truth Hamiltonian
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    
    # Edge index
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Create model
    model = HGNN(
        mu=mu,
        interaction_strength=1.0,
        interaction_radius=interaction_radius,
        hidden_dims=[64, 64, 32],
        integrator_type="stormer_verlet",
        dt=dt,
        create_graph=True,
        softplus_output=False,
        normalize_features=True,
        init_gain=1.0,
        use_dt_scaling=True,
    ).float()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gt_integrator = StormerVerlet(dt=dt, create_graph=False)
    
    best_loss = float('inf')
    best_state = None
    converged = False
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_grad_phi = 0.0
        epoch_grad_hbase = 0.0
        
        for batch_idx in range(batch_size):
            ic_seed = batch_idx + epoch * batch_size + seed * 10000
            q0, p0 = generate_ic(n_particles, mu, ic_seed)
            
            # Ground truth trajectory
            with torch.no_grad():
                q_true, p_true = q0.double().clone(), p0.double().clone()
                for _ in range(horizon):
                    q_true, p_true = gt_integrator.step(
                        q_true, p_true, gt_hamiltonian, edge_index
                    )
            
            # Model prediction
            q_pred, p_pred = model.forward(
                q0.float(), p0.float(), n_steps=horizon, return_trajectory=False
            )
            
            loss = torch.mean((q_pred - q_true.float())**2 + (p_pred - p_true.float())**2)
            
            # Alpha regularization
            alpha_reg = 1e-4 * model.hamiltonian.alpha.pow(2)
            total_loss = loss + alpha_reg
            
            total_loss.backward()
            epoch_loss += loss.item()
            
            # Log gradient norms
            if hasattr(model.hamiltonian, 'Phi'):
                phi_grad_norm = 0.0
                for p in model.hamiltonian.Phi.parameters():
                    if p.grad is not None:
                        phi_grad_norm += p.grad.norm().item() ** 2
                epoch_grad_phi += np.sqrt(phi_grad_norm)
            
            # Estimate H_base gradient scale (from position-kinetic coupling)
            hbase_grad_norm = torch.norm(p0.float()).item()  # Proxy for dH/dq scale
            epoch_grad_hbase += hbase_grad_norm
        
        # Update
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()
        
        avg_loss = epoch_loss / batch_size
        avg_grad_phi = epoch_grad_phi / batch_size
        avg_grad_hbase = epoch_grad_hbase / batch_size
        
        # Record metrics
        metrics.loss_history.append(avg_loss)
        metrics.alpha_history.append(model.hamiltonian.alpha.item())
        metrics.grad_phi_norms.append(avg_grad_phi)
        metrics.grad_hbase_norms.append(avg_grad_hbase)
        
        # Compute SNR
        snr = avg_grad_phi / (avg_grad_hbase + 1e-10)
        metrics.gradient_snr.append(snr)
        
        # Check for best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Check convergence
        if not converged and len(metrics.loss_history) > 20:
            recent = metrics.loss_history[-20:]
            if max(recent) - min(recent) < 0.1 * np.mean(recent):
                metrics.convergence_epoch = epoch
                converged = True
        
        # Check failure modes
        alpha = model.hamiltonian.alpha.item()
        if abs(alpha) > 1.0:
            metrics.alpha_diverged = True
            metrics.failure_reason = f"Alpha diverged: {alpha:.2e}"
        
        # Check Phi collapse
        with torch.no_grad():
            q_test, p_test = generate_ic(n_particles, mu, 9999)
            q_i = q_test.float()[edge_index[0]]
            p_i = p_test.float()[edge_index[0]]
            q_j = q_test.float()[edge_index[1]]
            p_j = p_test.float()[edge_index[1]]
            phi_out = model.hamiltonian.Phi(q_i, p_i, q_j, p_j)
            phi_norm = phi_out.abs().mean().item()
        
        if phi_norm < 1e-8:
            metrics.phi_collapsed = True
            metrics.failure_reason = f"Phi collapsed: {phi_norm:.2e}"
        
        if verbose and epoch % 30 == 0:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4e}, alpha={alpha:.4e}, "
                  f"SNR={snr:.4f}, |Phi|={phi_norm:.4f}")
    
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final evaluation
    metrics.final_loss = best_loss
    metrics.final_alpha = model.hamiltonian.alpha.item()
    
    # Compute final improvement
    improvement = evaluate_improvement(model, gt_hamiltonian, edge_index, 
                                        n_particles, mu, dt, horizon)
    metrics.final_improvement = improvement
    
    # Mark training as failed if no improvement
    if improvement < 0:
        metrics.training_failed = True
        if not metrics.failure_reason:
            metrics.failure_reason = "No improvement over baseline"
    
    return model, metrics


def evaluate_improvement(
    model: HGNN,
    gt_hamiltonian: MultiBodyHamiltonian,
    edge_index: torch.Tensor,
    n_particles: int,
    mu: float,
    dt: float,
    horizon: int,
    n_eval: int = 20,
) -> float:
    """Evaluate improvement over physics-only baseline."""
    model.eval()
    
    gt_integrator = StormerVerlet(dt=dt, create_graph=False)
    base_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=0.0)
    base_integrator = StormerVerlet(dt=dt, create_graph=False)
    
    model_errors = []
    base_errors = []
    
    with torch.no_grad():
        for i in range(n_eval):
            q0, p0 = generate_ic(n_particles, mu, 7000 + i)
            
            # Ground truth
            q_gt, p_gt = q0.double().clone(), p0.double().clone()
            for _ in range(horizon):
                q_gt, p_gt = gt_integrator.step(q_gt, p_gt, gt_hamiltonian, edge_index)
            
            # Model prediction
            q_model, p_model = model.forward(
                q0.float(), p0.float(), n_steps=horizon, return_trajectory=False
            )
            
            # Baseline
            q_base, p_base = q0.double().clone(), p0.double().clone()
            for _ in range(horizon):
                q_base, p_base = base_integrator.step(
                    q_base, p_base, base_hamiltonian, edge_index
                )
            
            model_err = torch.mean((q_model - q_gt.float())**2 + 
                                    (p_model - p_gt.float())**2).item()
            base_err = torch.mean((q_base.float() - q_gt.float())**2 + 
                                   (p_base.float() - p_gt.float())**2).item()
            
            model_errors.append(model_err)
            base_errors.append(base_err)
    
    avg_model = np.mean(model_errors)
    avg_base = np.mean(base_errors)
    
    improvement = 100 * (avg_base - avg_model) / (avg_base + 1e-10)
    return improvement


# ===========================================================================
# 1. LEARNABILITY THRESHOLD ANALYSIS
# ===========================================================================

def learnability_threshold_sweep(
    epsilon_range: Tuple[float, float] = (0.1, 300.0),
    n_points: int = 20,
    n_seeds: int = 3,
    n_particles: int = 10,
    dt: float = 60.0,
    horizon: int = 50,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Sweep epsilon to find critical learnability threshold.
    
    Returns:
        Dictionary with:
        - epsilon_values: tested values
        - metrics: TrainingMetrics for each (epsilon, seed)
        - scaling_law: fitted epsilon_crit and scaling
    """
    print("=" * 70)
    print("LEARNABILITY THRESHOLD ANALYSIS")
    print("=" * 70)
    
    if output_dir is None:
        output_dir = Path("results/scaling_laws")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dense logarithmic grid
    epsilon_values = np.logspace(
        np.log10(epsilon_range[0]),
        np.log10(epsilon_range[1]),
        n_points
    )
    
    print(f"Sweeping epsilon: {epsilon_values[0]:.2f} -> {epsilon_values[-1]:.2f}")
    print(f"Seeds: {n_seeds}, N={n_particles}, dt={dt}, T={horizon}")
    print("-" * 70)
    
    all_metrics = []
    results_by_epsilon = defaultdict(list)
    
    for eps in epsilon_values:
        print(f"\nEpsilon = {eps:.2f}")
        
        for seed in range(n_seeds):
            print(f"  Seed {seed}...", end=" ", flush=True)
            
            config = {
                'seed': seed,
                'epsilon': eps,
                'n_particles': n_particles,
                'dt': dt,
                'horizon': horizon,
                'n_epochs': 100,
                'batch_size': 8,
            }
            
            try:
                _, metrics = train_with_logging(config, verbose=False)
                results_by_epsilon[eps].append(metrics)
                all_metrics.append(metrics)
                print(f"imp={metrics.final_improvement:.1f}%, alpha={metrics.final_alpha:.2e}")
            except Exception as e:
                print(f"FAILED: {e}")
                metrics = TrainingMetrics(
                    seed=seed, epsilon=eps, n_particles=n_particles,
                    dt=dt, horizon=horizon, training_failed=True,
                    failure_reason=str(e)
                )
                all_metrics.append(metrics)
    
    # Analyze results
    scaling_result = analyze_epsilon_scaling(results_by_epsilon)
    
    # Save raw results
    raw_results = {
        'epsilon_values': epsilon_values.tolist(),
        'n_particles': n_particles,
        'dt': dt,
        'horizon': horizon,
        'n_seeds': n_seeds,
        'metrics': [asdict(m) for m in all_metrics],
        'scaling_law': asdict(scaling_result),
    }
    
    with open(output_dir / "epsilon_threshold.json", 'w') as f:
        json.dump(raw_results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("EPSILON SCALING LAW RESULTS")
    print("=" * 70)
    print(f"Critical epsilon: {scaling_result.critical_value:.2f}")
    print(f"Scaling exponent: {scaling_result.scaling_exponent:.3f}")
    print(f"R-squared: {scaling_result.r_squared:.4f}")
    
    return raw_results


def analyze_epsilon_scaling(results_by_epsilon: Dict) -> ScalingLawResult:
    """Fit scaling law to epsilon sweep results."""
    result = ScalingLawResult(parameter_name='epsilon')
    
    for eps in sorted(results_by_epsilon.keys()):
        metrics_list = results_by_epsilon[eps]
        improvements = [m.final_improvement for m in metrics_list if not m.training_failed]
        alphas = [m.final_alpha for m in metrics_list if not m.training_failed]
        
        if improvements:
            result.parameter_values.append(eps)
            result.mean_improvements.append(np.mean(improvements))
            result.std_improvements.append(np.std(improvements))
            result.mean_alphas.append(np.mean(alphas))
            result.std_alphas.append(np.std(alphas))
    
    # Find critical epsilon (transition from negative to positive improvement)
    eps_arr = np.array(result.parameter_values)
    imp_arr = np.array(result.mean_improvements)
    
    # Find first epsilon where improvement > 0
    positive_mask = imp_arr > 0
    if positive_mask.any():
        first_positive_idx = np.argmax(positive_mask)
        if first_positive_idx > 0:
            # Interpolate
            eps1 = eps_arr[first_positive_idx - 1]
            eps2 = eps_arr[first_positive_idx]
            imp1 = imp_arr[first_positive_idx - 1]
            imp2 = imp_arr[first_positive_idx]
            result.critical_value = eps1 - imp1 * (eps2 - eps1) / (imp2 - imp1 + 1e-10)
        else:
            result.critical_value = eps_arr[0]
    else:
        result.critical_value = eps_arr[-1]  # Never learned
    
    # Fit power law: improvement ~ (epsilon - eps_crit)^gamma
    try:
        mask = imp_arr > 0
        if mask.sum() >= 3:
            x = np.log(eps_arr[mask])
            y = np.log(np.abs(imp_arr[mask]) + 1e-10)
            coeffs = np.polyfit(x, y, 1)
            result.scaling_exponent = coeffs[0]
            result.scaling_coefficient = np.exp(coeffs[1])
            
            # R-squared
            y_pred = coeffs[0] * x + coeffs[1]
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            result.r_squared = 1 - ss_res / (ss_tot + 1e-10)
    except Exception:
        pass
    
    return result


# ===========================================================================
# 2. SYSTEM SIZE SCALING
# ===========================================================================

def system_size_scaling(
    n_values: List[int] = [10, 20, 40, 80, 160],
    n_seeds: int = 3,
    epsilon: float = 100.0,
    dt: float = 60.0,
    horizon: int = 50,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Analyze how correction learning scales with system size N.
    
    Tracks:
    - Training stability
    - Gradient magnitudes  
    - Alpha convergence
    - Improvement vs N
    
    Returns:
        Dictionary with scaling analysis results
    """
    print("=" * 70)
    print("SYSTEM SIZE SCALING ANALYSIS")
    print("=" * 70)
    
    if output_dir is None:
        output_dir = Path("results/scaling_laws")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"N values: {n_values}")
    print(f"Seeds: {n_seeds}, epsilon={epsilon}, dt={dt}, T={horizon}")
    print("-" * 70)
    
    all_metrics = []
    results_by_n = defaultdict(list)
    
    for n in n_values:
        print(f"\nN = {n}")
        
        for seed in range(n_seeds):
            print(f"  Seed {seed}...", end=" ", flush=True)
            
            # Reduce epochs for large N (O(N^2) complexity)
            n_epochs = max(50, 150 - n)
            
            config = {
                'seed': seed,
                'epsilon': epsilon,
                'n_particles': n,
                'dt': dt,
                'horizon': horizon,
                'n_epochs': n_epochs,
                'batch_size': max(4, 16 - n // 20),
            }
            
            try:
                start_time = time.time()
                _, metrics = train_with_logging(config, verbose=False)
                train_time = time.time() - start_time
                
                results_by_n[n].append(metrics)
                all_metrics.append(metrics)
                print(f"imp={metrics.final_improvement:.1f}%, alpha={metrics.final_alpha:.2e}, "
                      f"time={train_time:.1f}s")
            except Exception as e:
                print(f"FAILED: {e}")
                metrics = TrainingMetrics(
                    seed=seed, epsilon=epsilon, n_particles=n,
                    dt=dt, horizon=horizon, training_failed=True,
                    failure_reason=str(e)
                )
                all_metrics.append(metrics)
    
    # Analyze scaling
    scaling_result = analyze_n_scaling(results_by_n)
    
    # Save results
    raw_results = {
        'n_values': n_values,
        'epsilon': epsilon,
        'dt': dt,
        'horizon': horizon,
        'n_seeds': n_seeds,
        'metrics': [asdict(m) for m in all_metrics],
        'scaling_law': asdict(scaling_result),
    }
    
    with open(output_dir / "n_scaling.json", 'w') as f:
        json.dump(raw_results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("N SCALING LAW RESULTS")
    print("=" * 70)
    print(f"Alpha scaling exponent: {scaling_result.scaling_exponent:.3f}")
    print(f"Critical N (if any): {scaling_result.critical_value:.1f}")
    print(f"R-squared: {scaling_result.r_squared:.4f}")
    
    # Print table
    print("\nN\tMean Imp\tStd Imp\t\tMean Alpha")
    for i, n in enumerate(scaling_result.parameter_values):
        print(f"{n:.0f}\t{scaling_result.mean_improvements[i]:.2f}%\t\t"
              f"{scaling_result.std_improvements[i]:.2f}%\t\t"
              f"{scaling_result.mean_alphas[i]:.2e}")
    
    return raw_results


def analyze_n_scaling(results_by_n: Dict) -> ScalingLawResult:
    """Analyze scaling with system size N."""
    result = ScalingLawResult(parameter_name='N')
    
    for n in sorted(results_by_n.keys()):
        metrics_list = results_by_n[n]
        improvements = [m.final_improvement for m in metrics_list if not m.training_failed]
        alphas = [abs(m.final_alpha) for m in metrics_list if not m.training_failed]
        
        if improvements:
            result.parameter_values.append(n)
            result.mean_improvements.append(np.mean(improvements))
            result.std_improvements.append(np.std(improvements))
            result.mean_alphas.append(np.mean(alphas))
            result.std_alphas.append(np.std(alphas))
    
    # Find N_crit (where improvement drops below 0)
    n_arr = np.array(result.parameter_values)
    imp_arr = np.array(result.mean_improvements)
    
    negative_mask = imp_arr < 0
    if negative_mask.any():
        first_negative_idx = np.argmax(negative_mask)
        if first_negative_idx > 0:
            n1 = n_arr[first_negative_idx - 1]
            n2 = n_arr[first_negative_idx]
            imp1 = imp_arr[first_negative_idx - 1]
            imp2 = imp_arr[first_negative_idx]
            result.critical_value = n1 - imp1 * (n2 - n1) / (imp2 - imp1 + 1e-10)
        else:
            result.critical_value = n_arr[0]
    else:
        result.critical_value = float('inf')  # Never failed
    
    # Fit alpha scaling: alpha ~ N^gamma
    try:
        alpha_arr = np.array(result.mean_alphas)
        mask = alpha_arr > 0
        if mask.sum() >= 3:
            x = np.log(n_arr[mask])
            y = np.log(alpha_arr[mask])
            coeffs = np.polyfit(x, y, 1)
            result.scaling_exponent = coeffs[0]
            result.scaling_coefficient = np.exp(coeffs[1])
            
            y_pred = coeffs[0] * x + coeffs[1]
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            result.r_squared = 1 - ss_res / (ss_tot + 1e-10)
    except Exception:
        pass
    
    return result


# ===========================================================================
# 3. LYAPUNOV EXPONENT ANALYSIS
# ===========================================================================

def estimate_lyapunov_exponent(
    model: Optional[HGNN],
    n_particles: int = 10,
    epsilon: float = 100.0,
    dt: float = 60.0,
    n_steps: int = 500,
    n_samples: int = 10,
    perturbation: float = 1e-6,
    mu: float = 398600.4418,
) -> Dict[str, float]:
    """
    Estimate largest Lyapunov exponent via perturbation growth.
    
    Args:
        model: H-GNN model (None for ground truth baseline)
        n_particles: Number of particles
        epsilon: Interaction strength
        dt: Time step
        n_steps: Number of integration steps
        n_samples: Number of sample trajectories
        perturbation: Initial perturbation magnitude
        mu: Central body parameter
    
    Returns:
        Dictionary with Lyapunov exponent estimates
    """
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Ground truth
    gt_hamiltonian = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    gt_integrator = StormerVerlet(dt=dt, create_graph=False)
    
    lyapunov_estimates = []
    
    for sample in range(n_samples):
        q0, p0 = generate_ic(n_particles, mu, 1000 + sample)
        
        # Add small perturbation
        q0_pert = q0 + perturbation * torch.randn_like(q0)
        p0_pert = p0 + perturbation * torch.randn_like(p0)
        
        initial_dist = torch.sqrt(torch.sum((q0 - q0_pert)**2 + (p0 - p0_pert)**2)).item()
        
        # Evolve both trajectories
        with torch.no_grad():
            q, p = q0.double().clone(), p0.double().clone()
            q_pert, p_pert = q0_pert.double().clone(), p0_pert.double().clone()
            
            divergence_log = []
            
            for step in range(n_steps):
                if model is not None:
                    # Use model
                    q_new, p_new = model.forward(
                        q.float(), p.float(), n_steps=1, return_trajectory=False
                    )
                    q_pert_new, p_pert_new = model.forward(
                        q_pert.float(), p_pert.float(), n_steps=1, return_trajectory=False
                    )
                    q, p = q_new.double(), p_new.double()
                    q_pert, p_pert = q_pert_new.double(), p_pert_new.double()
                else:
                    # Use ground truth
                    q, p = gt_integrator.step(q, p, gt_hamiltonian, edge_index)
                    q_pert, p_pert = gt_integrator.step(
                        q_pert, p_pert, gt_hamiltonian, edge_index
                    )
                
                dist = torch.sqrt(torch.sum((q - q_pert)**2 + (p - p_pert)**2)).item()
                divergence_log.append(np.log(dist / initial_dist + 1e-16))
        
        # Fit linear growth: log(d(t)) = lambda * t + const
        t = np.arange(1, n_steps + 1) * dt
        log_d = np.array(divergence_log)
        
        try:
            coeffs = np.polyfit(t, log_d, 1)
            lyapunov_estimates.append(coeffs[0])
        except Exception:
            pass
    
    return {
        'lyapunov_mean': np.mean(lyapunov_estimates) if lyapunov_estimates else 0.0,
        'lyapunov_std': np.std(lyapunov_estimates) if lyapunov_estimates else 0.0,
        'n_samples': len(lyapunov_estimates),
    }


def chaos_analysis(
    n_particles: int = 10,
    epsilon: float = 100.0,
    dt: float = 60.0,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Compare Lyapunov exponents for different models.
    
    Tests:
    - Ground truth (full Hamiltonian)
    - Physics-only baseline (epsilon=0)
    - Trained H-GNN
    - Vanilla GNN (non-symplectic, for comparison)
    """
    print("=" * 70)
    print("CHAOS AND LYAPUNOV ANALYSIS")
    print("=" * 70)
    
    if output_dir is None:
        output_dir = Path("results/scaling_laws")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mu = 398600.4418
    
    results = {}
    
    # 1. Ground truth
    print("\n1. Ground truth (full Hamiltonian)...")
    results['ground_truth'] = estimate_lyapunov_exponent(
        model=None, n_particles=n_particles, epsilon=epsilon, dt=dt,
        n_steps=300, n_samples=10, mu=mu
    )
    print(f"   Lyapunov: {results['ground_truth']['lyapunov_mean']:.6f} +/- "
          f"{results['ground_truth']['lyapunov_std']:.6f}")
    
    # 2. Physics-only baseline
    print("\n2. Physics-only baseline (epsilon=0)...")
    results['physics_only'] = estimate_lyapunov_exponent(
        model=None, n_particles=n_particles, epsilon=0.0, dt=dt,
        n_steps=300, n_samples=10, mu=mu
    )
    print(f"   Lyapunov: {results['physics_only']['lyapunov_mean']:.6f} +/- "
          f"{results['physics_only']['lyapunov_std']:.6f}")
    
    # 3. Train H-GNN
    print("\n3. Training H-GNN model...")
    config = {
        'seed': 42,
        'epsilon': epsilon,
        'n_particles': n_particles,
        'dt': dt,
        'horizon': 50,
        'n_epochs': 100,
    }
    hgnn_model, hgnn_metrics = train_with_logging(config, verbose=False)
    print(f"   H-GNN improvement: {hgnn_metrics.final_improvement:.2f}%")
    
    print("   Estimating Lyapunov exponent...")
    results['hgnn'] = estimate_lyapunov_exponent(
        model=hgnn_model, n_particles=n_particles, epsilon=epsilon, dt=dt,
        n_steps=300, n_samples=10, mu=mu
    )
    print(f"   Lyapunov: {results['hgnn']['lyapunov_mean']:.6f} +/- "
          f"{results['hgnn']['lyapunov_std']:.6f}")
    
    # Save results
    with open(output_dir / "lyapunov_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("LYAPUNOV ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Ground Truth:  {results['ground_truth']['lyapunov_mean']:.6f}")
    print(f"Physics Only:  {results['physics_only']['lyapunov_mean']:.6f}")
    print(f"H-GNN:         {results['hgnn']['lyapunov_mean']:.6f}")
    
    # Check if H-GNN reduces divergence
    if results['hgnn']['lyapunov_mean'] < results['physics_only']['lyapunov_mean']:
        print("\n>>> H-GNN REDUCES exponential divergence compared to physics-only")
    else:
        print("\n>>> H-GNN does NOT reduce exponential divergence")
    
    return results


# ===========================================================================
# 4. BACKWARD ERROR DT VALIDATION
# ===========================================================================

def backward_error_dt_validation(
    dt_values: List[float] = [15.0, 30.0, 60.0, 120.0],
    n_particles: int = 10,
    epsilon: float = 100.0,
    n_seeds: int = 3,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Validate that learned correction scales as dt^2.
    
    Tests:
    - Train models at multiple dt values
    - Evaluate MSE scaling with dt
    - Fit power law to verify dt^2 behavior
    """
    print("=" * 70)
    print("BACKWARD ERROR DT VALIDATION")
    print("=" * 70)
    
    if output_dir is None:
        output_dir = Path("results/scaling_laws")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"dt values: {dt_values}")
    print(f"N={n_particles}, epsilon={epsilon}, seeds={n_seeds}")
    print("-" * 70)
    
    mu = 398600.4418
    all_metrics = []
    results_by_dt = defaultdict(list)
    
    for dt in dt_values:
        print(f"\ndt = {dt}")
        
        for seed in range(n_seeds):
            print(f"  Seed {seed}...", end=" ", flush=True)
            
            # Adjust horizon to cover same physical time
            physical_time = 3000.0  # seconds
            horizon = int(physical_time / dt)
            
            config = {
                'seed': seed,
                'epsilon': epsilon,
                'n_particles': n_particles,
                'dt': dt,
                'horizon': horizon,
                'n_epochs': 100,
                'batch_size': 8,
            }
            
            try:
                _, metrics = train_with_logging(config, verbose=False)
                results_by_dt[dt].append(metrics)
                all_metrics.append(metrics)
                print(f"imp={metrics.final_improvement:.1f}%, alpha={metrics.final_alpha:.2e}")
            except Exception as e:
                print(f"FAILED: {e}")
    
    # Analyze dt scaling
    scaling_result = analyze_dt_scaling(results_by_dt)
    
    # Save results
    raw_results = {
        'dt_values': dt_values,
        'n_particles': n_particles,
        'epsilon': epsilon,
        'n_seeds': n_seeds,
        'metrics': [asdict(m) for m in all_metrics],
        'scaling_law': asdict(scaling_result),
    }
    
    with open(output_dir / "dt_scaling.json", 'w') as f:
        json.dump(raw_results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("DT SCALING LAW RESULTS")
    print("=" * 70)
    print(f"Alpha scaling exponent: {scaling_result.scaling_exponent:.3f}")
    print(f"  (Expected: ~2.0 for dt^2 scaling)")
    print(f"R-squared: {scaling_result.r_squared:.4f}")
    
    return raw_results


def analyze_dt_scaling(results_by_dt: Dict) -> ScalingLawResult:
    """Analyze scaling with timestep dt."""
    result = ScalingLawResult(parameter_name='dt')
    
    for dt in sorted(results_by_dt.keys()):
        metrics_list = results_by_dt[dt]
        improvements = [m.final_improvement for m in metrics_list if not m.training_failed]
        alphas = [abs(m.final_alpha) for m in metrics_list if not m.training_failed]
        
        if improvements:
            result.parameter_values.append(dt)
            result.mean_improvements.append(np.mean(improvements))
            result.std_improvements.append(np.std(improvements))
            result.mean_alphas.append(np.mean(alphas))
            result.std_alphas.append(np.std(alphas))
    
    # Fit alpha ~ dt^gamma (should be ~2 for backward error alignment)
    try:
        dt_arr = np.array(result.parameter_values)
        alpha_arr = np.array(result.mean_alphas)
        
        mask = alpha_arr > 0
        if mask.sum() >= 3:
            x = np.log(dt_arr[mask])
            y = np.log(alpha_arr[mask])
            coeffs = np.polyfit(x, y, 1)
            result.scaling_exponent = coeffs[0]
            result.scaling_coefficient = np.exp(coeffs[1])
            
            y_pred = coeffs[0] * x + coeffs[1]
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            result.r_squared = 1 - ss_res / (ss_tot + 1e-10)
    except Exception:
        pass
    
    return result


# ===========================================================================
# 5. FAILURE MODE CHARACTERIZATION
# ===========================================================================

def characterize_failure_modes(
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Systematically identify failure conditions.
    
    Tests failure modes:
    1. Phi collapse: when does Phi output go to zero?
    2. Alpha divergence: when does alpha explode?
    3. N-scaling failure: when does learning fail with large N?
    4. Epsilon failure: when is signal too weak?
    """
    print("=" * 70)
    print("FAILURE MODE CHARACTERIZATION")
    print("=" * 70)
    
    if output_dir is None:
        output_dir = Path("results/scaling_laws")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    failure_results = {
        'phi_collapse': [],
        'alpha_divergence': [],
        'n_scaling_failure': [],
        'epsilon_failure': [],
    }
    
    # 1. Test for Phi collapse (small epsilon)
    print("\n1. Testing Phi collapse (small epsilon)...")
    for eps in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        config = {
            'seed': 42,
            'epsilon': eps,
            'n_particles': 10,
            'dt': 60.0,
            'horizon': 20,
            'n_epochs': 50,
        }
        try:
            _, metrics = train_with_logging(config, verbose=False)
            failure_results['phi_collapse'].append({
                'epsilon': eps,
                'phi_collapsed': metrics.phi_collapsed,
                'improvement': metrics.final_improvement,
                'final_alpha': metrics.final_alpha,
            })
            status = "COLLAPSED" if metrics.phi_collapsed else "OK"
            print(f"  eps={eps:.2f}: {status}, imp={metrics.final_improvement:.1f}%")
        except Exception as e:
            print(f"  eps={eps:.2f}: ERROR - {e}")
    
    # 2. Test for alpha divergence (high learning rate)
    print("\n2. Testing alpha divergence (high LR)...")
    for lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        config = {
            'seed': 42,
            'epsilon': 100.0,
            'n_particles': 10,
            'dt': 60.0,
            'horizon': 50,
            'n_epochs': 50,
            'lr': lr,
        }
        try:
            _, metrics = train_with_logging(config, verbose=False)
            failure_results['alpha_divergence'].append({
                'lr': lr,
                'alpha_diverged': metrics.alpha_diverged,
                'final_alpha': metrics.final_alpha,
                'improvement': metrics.final_improvement,
            })
            status = "DIVERGED" if metrics.alpha_diverged else "OK"
            print(f"  lr={lr:.0e}: {status}, alpha={metrics.final_alpha:.2e}")
        except Exception as e:
            print(f"  lr={lr:.0e}: ERROR - {e}")
    
    # 3. Test N-scaling failure
    print("\n3. Testing N-scaling failure...")
    for n in [10, 25, 50, 80, 100]:
        config = {
            'seed': 42,
            'epsilon': 100.0,
            'n_particles': n,
            'dt': 60.0,
            'horizon': 30,
            'n_epochs': 50,
        }
        try:
            start = time.time()
            _, metrics = train_with_logging(config, verbose=False)
            train_time = time.time() - start
            failure_results['n_scaling_failure'].append({
                'n': n,
                'training_failed': metrics.training_failed,
                'improvement': metrics.final_improvement,
                'training_time': train_time,
            })
            status = "FAILED" if metrics.training_failed else "OK"
            print(f"  N={n}: {status}, imp={metrics.final_improvement:.1f}%, time={train_time:.1f}s")
        except Exception as e:
            print(f"  N={n}: ERROR - {e}")
    
    # 4. Test epsilon failure boundary
    print("\n4. Testing epsilon failure boundary...")
    for eps in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        config = {
            'seed': 42,
            'epsilon': eps,
            'n_particles': 10,
            'dt': 60.0,
            'horizon': 50,
            'n_epochs': 80,
        }
        try:
            _, metrics = train_with_logging(config, verbose=False)
            failure_results['epsilon_failure'].append({
                'epsilon': eps,
                'training_failed': metrics.training_failed,
                'improvement': metrics.final_improvement,
                'gradient_snr': np.mean(metrics.gradient_snr) if metrics.gradient_snr else 0,
            })
            status = "FAILED" if metrics.training_failed else "OK"
            print(f"  eps={eps:.2f}: {status}, imp={metrics.final_improvement:.1f}%")
        except Exception as e:
            print(f"  eps={eps:.2f}: ERROR - {e}")
    
    # Save results
    with open(output_dir / "failure_modes.json", 'w') as f:
        json.dump(failure_results, f, indent=2)
    
    # Summarize
    print("\n" + "=" * 70)
    print("FAILURE MODE SUMMARY")
    print("=" * 70)
    
    # Find boundaries
    phi_collapse_eps = min(
        [r['epsilon'] for r in failure_results['phi_collapse'] if r['phi_collapsed']],
        default=float('inf')
    )
    
    alpha_diverge_lr = max(
        [r['lr'] for r in failure_results['alpha_divergence'] if r['alpha_diverged']],
        default=0.0
    )
    
    n_fail = min(
        [r['n'] for r in failure_results['n_scaling_failure'] if r['training_failed']],
        default=float('inf')
    )
    
    eps_fail = max(
        [r['epsilon'] for r in failure_results['epsilon_failure'] if r['training_failed']],
        default=0.0
    )
    
    print(f"Phi collapse boundary: epsilon < {phi_collapse_eps:.2f}")
    print(f"Alpha divergence boundary: lr > {alpha_diverge_lr:.1e}")
    print(f"N-scaling failure boundary: N > {n_fail}")
    print(f"Epsilon failure boundary: epsilon < {eps_fail:.2f}")
    
    return failure_results


# ===========================================================================
# UNIFIED EXPERIMENT RUNNER
# ===========================================================================

def run_all_scaling_experiments(
    output_dir: Path = None,
    run_epsilon_sweep: bool = True,
    run_n_scaling: bool = True,
    run_lyapunov: bool = True,
    run_dt_validation: bool = True,
    run_failure_modes: bool = True,
) -> Dict[str, Any]:
    """
    Run all scaling law discovery experiments.
    
    Returns comprehensive results for analysis.
    """
    if output_dir is None:
        output_dir = Path("results/scaling_laws")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SCALING LAW DISCOVERY FRAMEWORK")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    all_results = {}
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. Epsilon threshold sweep
    if run_epsilon_sweep:
        print("\n\n>>> EXPERIMENT 1: LEARNABILITY THRESHOLD <<<\n")
        all_results['epsilon_threshold'] = learnability_threshold_sweep(
            epsilon_range=(0.1, 300.0),
            n_points=15,
            n_seeds=3,
            output_dir=output_dir,
        )
    
    # 2. System size scaling
    if run_n_scaling:
        print("\n\n>>> EXPERIMENT 2: SYSTEM SIZE SCALING <<<\n")
        all_results['n_scaling'] = system_size_scaling(
            n_values=[10, 20, 40, 80],
            n_seeds=3,
            output_dir=output_dir,
        )
    
    # 3. Lyapunov analysis
    if run_lyapunov:
        print("\n\n>>> EXPERIMENT 3: CHAOS AND LYAPUNOV <<<\n")
        all_results['lyapunov'] = chaos_analysis(
            n_particles=10,
            epsilon=100.0,
            output_dir=output_dir,
        )
    
    # 4. Backward error dt validation
    if run_dt_validation:
        print("\n\n>>> EXPERIMENT 4: BACKWARD ERROR DT VALIDATION <<<\n")
        all_results['dt_validation'] = backward_error_dt_validation(
            dt_values=[15.0, 30.0, 60.0, 120.0],
            n_seeds=3,
            output_dir=output_dir,
        )
    
    # 5. Failure mode characterization
    if run_failure_modes:
        print("\n\n>>> EXPERIMENT 5: FAILURE MODE CHARACTERIZATION <<<\n")
        all_results['failure_modes'] = characterize_failure_modes(
            output_dir=output_dir,
        )
    
    # Save master results
    with open(output_dir / f"master_results_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate summary report
    generate_scaling_report(all_results, output_dir)
    
    return all_results


def generate_scaling_report(results: Dict, output_dir: Path):
    """Generate markdown report of scaling law findings."""
    report = []
    report.append("# Scaling Law Discovery Results")
    report.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Epsilon scaling
    if 'epsilon_threshold' in results:
        sl = results['epsilon_threshold'].get('scaling_law', {})
        report.append("## 1. Learnability Threshold")
        report.append(f"- **Critical epsilon**: {sl.get('critical_value', 'N/A'):.2f}")
        report.append(f"- **Scaling exponent**: {sl.get('scaling_exponent', 'N/A'):.3f}")
        report.append(f"- **R-squared**: {sl.get('r_squared', 'N/A'):.4f}")
        report.append("")
    
    # N scaling
    if 'n_scaling' in results:
        sl = results['n_scaling'].get('scaling_law', {})
        report.append("## 2. System Size Scaling")
        report.append(f"- **Alpha scaling exponent**: {sl.get('scaling_exponent', 'N/A'):.3f}")
        report.append(f"- **Critical N**: {sl.get('critical_value', 'N/A')}")
        report.append(f"- **R-squared**: {sl.get('r_squared', 'N/A'):.4f}")
        report.append("")
    
    # Lyapunov
    if 'lyapunov' in results:
        lyap = results['lyapunov']
        report.append("## 3. Chaos Analysis")
        report.append(f"- **Ground truth Lyapunov**: {lyap.get('ground_truth', {}).get('lyapunov_mean', 'N/A'):.6f}")
        report.append(f"- **Physics-only Lyapunov**: {lyap.get('physics_only', {}).get('lyapunov_mean', 'N/A'):.6f}")
        report.append(f"- **H-GNN Lyapunov**: {lyap.get('hgnn', {}).get('lyapunov_mean', 'N/A'):.6f}")
        report.append("")
    
    # dt scaling
    if 'dt_validation' in results:
        sl = results['dt_validation'].get('scaling_law', {})
        report.append("## 4. Backward Error dt Validation")
        report.append(f"- **Alpha ~ dt^{sl.get('scaling_exponent', 'N/A'):.2f}**")
        report.append(f"- **Expected**: dt^2.0 for backward error alignment")
        report.append(f"- **R-squared**: {sl.get('r_squared', 'N/A'):.4f}")
        report.append("")
    
    # Failure modes
    if 'failure_modes' in results:
        fm = results['failure_modes']
        report.append("## 5. Failure Mode Boundaries")
        
        phi_collapse = [r for r in fm.get('phi_collapse', []) if r.get('phi_collapsed')]
        if phi_collapse:
            report.append(f"- **Phi collapse**: epsilon < {min(r['epsilon'] for r in phi_collapse):.2f}")
        
        alpha_div = [r for r in fm.get('alpha_divergence', []) if r.get('alpha_diverged')]
        if alpha_div:
            report.append(f"- **Alpha divergence**: lr > {max(r['lr'] for r in alpha_div):.1e}")
        
        report.append("")
    
    # Save report
    with open(output_dir / "SCALING_LAWS_REPORT.md", 'w') as f:
        f.write('\n'.join(report))
    
    print("\n" + "=" * 70)
    print("Report saved to:", output_dir / "SCALING_LAWS_REPORT.md")


# ===========================================================================
# COMMAND LINE INTERFACE
# ===========================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scaling Law Discovery for H-GNN")
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--epsilon', action='store_true', help='Run epsilon sweep')
    parser.add_argument('--n-scaling', action='store_true', help='Run N scaling')
    parser.add_argument('--lyapunov', action='store_true', help='Run Lyapunov analysis')
    parser.add_argument('--dt', action='store_true', help='Run dt validation')
    parser.add_argument('--failure', action='store_true', help='Run failure mode analysis')
    parser.add_argument('--output', type=str, default='results/scaling_laws',
                        help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.all or not any([args.epsilon, args.n_scaling, args.lyapunov, args.dt, args.failure]):
        # Run all by default
        run_all_scaling_experiments(output_dir=output_dir)
    else:
        # Run selected experiments
        run_all_scaling_experiments(
            output_dir=output_dir,
            run_epsilon_sweep=args.epsilon,
            run_n_scaling=args.n_scaling,
            run_lyapunov=args.lyapunov,
            run_dt_validation=args.dt,
            run_failure_modes=args.failure,
        )
