"""
Evaluation metrics for Hamiltonian systems.

This module provides metrics for validating:
1. Physics preservation (energy, angular momentum conservation)
2. Long-horizon stability
3. ML performance (rare-regime generalization)

All metrics are designed to be logged to wandb for experiment tracking.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class PhysicsMetrics:
    """Container for physics-based validation metrics."""
    energy_drift: torch.Tensor          # Relative energy change over time
    energy_drift_mean: float            # Mean relative energy drift
    energy_drift_max: float             # Maximum relative energy drift
    energy_oscillation: float           # Standard deviation of energy (bounded for symplectic)
    angular_momentum_drift: torch.Tensor  # Relative L change over time
    angular_momentum_drift_mean: float
    angular_momentum_drift_max: float
    eccentricity_drift: Optional[float] = None  # For Keplerian orbits


@dataclass
class StabilityMetrics:
    """Container for numerical stability metrics."""
    trajectory_bounded: bool            # Did trajectory stay bounded?
    max_position_norm: float            # Maximum |q| reached
    max_momentum_norm: float            # Maximum |p| reached
    nan_detected: bool                  # Any NaN values?
    inf_detected: bool                  # Any Inf values?
    final_time_reached: bool            # Did integration complete?


@dataclass
class MLMetrics:
    """Container for ML performance metrics."""
    rare_regime_recall: float           # Recall on rare events
    rare_regime_precision: float        # Precision on rare events
    false_negative_rate: float          # Critical for safety
    calibration_error: float            # Expected calibration error
    prediction_mse: float               # Mean squared error
    prediction_mae: float               # Mean absolute error


class MetricsComputer:
    """
    Compute physics and ML metrics for H-GNN evaluation.
    
    This class provides methods to evaluate:
    - Conservation law violations (energy, angular momentum)
    - Long-horizon stability
    - Rare-regime generalization performance
    
    Args:
        mu: Gravitational parameter for orbital mechanics
        device: Torch device
    """
    
    def __init__(
        self,
        mu: float = 398600.4418,
        device: str = "cpu",
    ):
        self.mu = mu
        self.device = device
    
    def compute_energy(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        hamiltonian: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Compute total energy (Hamiltonian).
        
        Args:
            q: Positions (..., 3)
            p: Momenta (..., 3)
            hamiltonian: Optional Hamiltonian module, else uses two-body
            
        Returns:
            Energy tensor
        """
        if hamiltonian is not None:
            with torch.no_grad():
                return hamiltonian(q, p, sum_over_particles=False)
        
        # Default two-body Hamiltonian
        T = 0.5 * torch.sum(p ** 2, dim=-1)
        r = torch.norm(q, dim=-1)
        V = -self.mu / (r + 1e-10)
        return T + V
    
    def compute_angular_momentum(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute angular momentum L = q × p.
        
        Args:
            q: Positions (..., 3)
            p: Momenta (..., 3)
            
        Returns:
            Angular momentum vector (..., 3)
        """
        return torch.cross(q, p, dim=-1)
    
    def compute_angular_momentum_magnitude(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute |L| = |q × p|."""
        L = self.compute_angular_momentum(q, p)
        return torch.norm(L, dim=-1)
    
    def compute_physics_metrics(
        self,
        q_trajectory: torch.Tensor,
        p_trajectory: torch.Tensor,
        hamiltonian: Optional[nn.Module] = None,
    ) -> PhysicsMetrics:
        """
        Compute physics conservation metrics from trajectory.
        
        Args:
            q_trajectory: Position trajectory (T, N, 3) or (T, 3)
            p_trajectory: Momentum trajectory (T, N, 3) or (T, 3)
            hamiltonian: Optional Hamiltonian module
            
        Returns:
            PhysicsMetrics dataclass with conservation violations
        """
        T = q_trajectory.shape[0]
        
        # Compute energy at each timestep
        energies = []
        L_magnitudes = []
        
        for t in range(T):
            q_t = q_trajectory[t]
            p_t = p_trajectory[t]
            
            # Sum over particles if needed
            E_t = self.compute_energy(q_t, p_t, hamiltonian)
            if E_t.dim() > 0:
                E_t = E_t.sum()
            energies.append(E_t.item())
            
            # Angular momentum
            L_t = self.compute_angular_momentum_magnitude(q_t, p_t)
            if L_t.dim() > 0:
                L_t = L_t.sum()
            L_magnitudes.append(L_t.item())
        
        energies = torch.tensor(energies)
        L_magnitudes = torch.tensor(L_magnitudes)
        
        # Initial values
        E0 = energies[0]
        L0 = L_magnitudes[0]
        
        # Relative drifts
        energy_drift = (energies - E0) / (torch.abs(E0) + 1e-10)
        L_drift = (L_magnitudes - L0) / (torch.abs(L0) + 1e-10)
        
        return PhysicsMetrics(
            energy_drift=energy_drift,
            energy_drift_mean=torch.abs(energy_drift).mean().item(),
            energy_drift_max=torch.abs(energy_drift).max().item(),
            energy_oscillation=energy_drift.std().item(),
            angular_momentum_drift=L_drift,
            angular_momentum_drift_mean=torch.abs(L_drift).mean().item(),
            angular_momentum_drift_max=torch.abs(L_drift).max().item(),
        )
    
    def compute_stability_metrics(
        self,
        q_trajectory: torch.Tensor,
        p_trajectory: torch.Tensor,
        position_bound: float = 1e8,
        momentum_bound: float = 1e6,
    ) -> StabilityMetrics:
        """
        Compute numerical stability metrics.
        
        Args:
            q_trajectory: Position trajectory
            p_trajectory: Momentum trajectory
            position_bound: Maximum allowed position norm
            momentum_bound: Maximum allowed momentum norm
            
        Returns:
            StabilityMetrics dataclass
        """
        # Check for NaN/Inf
        nan_q = torch.isnan(q_trajectory).any().item()
        nan_p = torch.isnan(p_trajectory).any().item()
        inf_q = torch.isinf(q_trajectory).any().item()
        inf_p = torch.isinf(p_trajectory).any().item()
        
        # Maximum norms
        max_q = torch.norm(q_trajectory, dim=-1).max().item()
        max_p = torch.norm(p_trajectory, dim=-1).max().item()
        
        # Boundedness check
        bounded = max_q < position_bound and max_p < momentum_bound
        
        return StabilityMetrics(
            trajectory_bounded=bounded and not (nan_q or nan_p or inf_q or inf_p),
            max_position_norm=max_q,
            max_momentum_norm=max_p,
            nan_detected=nan_q or nan_p,
            inf_detected=inf_q or inf_p,
            final_time_reached=not (nan_q or nan_p or inf_q or inf_p),
        )
    
    def compute_rare_regime_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        is_rare: torch.Tensor,
        threshold: float = 0.5,
    ) -> MLMetrics:
        """
        Compute ML metrics focused on rare regime generalization.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            is_rare: Boolean mask indicating rare regime samples
            threshold: Classification threshold for binary predictions
            
        Returns:
            MLMetrics dataclass with rare-regime performance
        """
        # Overall MSE and MAE
        mse = torch.mean((predictions - targets) ** 2).item()
        mae = torch.mean(torch.abs(predictions - targets)).item()
        
        # Rare regime metrics (classification perspective)
        if is_rare.sum() > 0:
            rare_preds = predictions[is_rare]
            rare_targets = targets[is_rare]
            
            # For binary classification (e.g., conjunction detection)
            if rare_targets.max() <= 1:
                pred_binary = (predictions > threshold).float()
                target_binary = targets.float()
                
                # True positives, false positives, false negatives
                tp = ((pred_binary == 1) & (target_binary == 1)).sum().float()
                fp = ((pred_binary == 1) & (target_binary == 0)).sum().float()
                fn = ((pred_binary == 0) & (target_binary == 1)).sum().float()
                
                precision = (tp / (tp + fp + 1e-10)).item()
                recall = (tp / (tp + fn + 1e-10)).item()
                fnr = (fn / (fn + tp + 1e-10)).item()
            else:
                # For regression, use threshold-based metrics
                precision = 1.0 - torch.mean(torch.abs(rare_preds - rare_targets)).item()
                recall = precision  # Simplified
                fnr = 0.0
        else:
            precision = 0.0
            recall = 0.0
            fnr = 1.0
        
        # Expected calibration error (simplified)
        ece = self._compute_ece(predictions, targets, n_bins=10)
        
        return MLMetrics(
            rare_regime_recall=recall,
            rare_regime_precision=precision,
            false_negative_rate=fnr,
            calibration_error=ece,
            prediction_mse=mse,
            prediction_mae=mae,
        )
    
    def _compute_ece(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        if predictions.dim() > 1:
            predictions = predictions.flatten()
        if targets.dim() > 1:
            targets = targets.flatten()
        
        # Normalize predictions to [0, 1] for calibration
        pred_min, pred_max = predictions.min(), predictions.max()
        if pred_max > pred_min:
            predictions_norm = (predictions - pred_min) / (pred_max - pred_min)
        else:
            return 0.0
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (predictions_norm >= bin_boundaries[i]) & (predictions_norm < bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                avg_pred = predictions[in_bin].mean().item()
                avg_target = targets[in_bin].mean().item()
                ece += (in_bin.sum().item() / len(predictions)) * abs(avg_pred - avg_target)
        
        return ece


def evaluate_integrator_comparison(
    q0: torch.Tensor,
    p0: torch.Tensor,
    hamiltonian: nn.Module,
    integrators: Dict[str, nn.Module],
    n_steps: int,
    dt: float,
) -> Dict[str, Dict[str, Union[PhysicsMetrics, StabilityMetrics]]]:
    """
    Compare multiple integrators on the same initial conditions.
    
    This is the key validation for Phase 1: proving symplectic
    integrators preserve structure better than non-symplectic ones.
    
    Args:
        q0: Initial positions
        p0: Initial momenta
        hamiltonian: Hamiltonian module
        integrators: Dict mapping names to integrator modules
        n_steps: Number of integration steps
        dt: Time step
        
    Returns:
        Dict with physics and stability metrics for each integrator
    """
    metrics_computer = MetricsComputer()
    results = {}
    
    for name, integrator in integrators.items():
        # Integrate
        integrator.dt = torch.tensor(dt, dtype=torch.float64)
        q_traj, p_traj = integrator.integrate(
            q0.clone(),
            p0.clone(),
            hamiltonian,
            n_steps,
            store_trajectory=True,
        )
        
        # Compute metrics
        physics = metrics_computer.compute_physics_metrics(q_traj, p_traj, hamiltonian)
        stability = metrics_computer.compute_stability_metrics(q_traj, p_traj)
        
        results[name] = {
            "physics": physics,
            "stability": stability,
            "q_trajectory": q_traj,
            "p_trajectory": p_traj,
        }
    
    return results


def log_metrics_to_wandb(
    metrics: Union[PhysicsMetrics, StabilityMetrics, MLMetrics],
    prefix: str = "",
    step: Optional[int] = None,
):
    """
    Log metrics to wandb.
    
    Args:
        metrics: Metrics dataclass to log
        prefix: Prefix for metric names
        step: Optional step number
    """
    try:
        import wandb
    except ImportError:
        warnings.warn("wandb not installed, skipping logging")
        return
    
    log_dict = {}
    
    if isinstance(metrics, PhysicsMetrics):
        log_dict[f"{prefix}energy_drift_mean"] = metrics.energy_drift_mean
        log_dict[f"{prefix}energy_drift_max"] = metrics.energy_drift_max
        log_dict[f"{prefix}energy_oscillation"] = metrics.energy_oscillation
        log_dict[f"{prefix}angular_momentum_drift_mean"] = metrics.angular_momentum_drift_mean
        log_dict[f"{prefix}angular_momentum_drift_max"] = metrics.angular_momentum_drift_max
    
    elif isinstance(metrics, StabilityMetrics):
        log_dict[f"{prefix}trajectory_bounded"] = int(metrics.trajectory_bounded)
        log_dict[f"{prefix}max_position_norm"] = metrics.max_position_norm
        log_dict[f"{prefix}max_momentum_norm"] = metrics.max_momentum_norm
        log_dict[f"{prefix}nan_detected"] = int(metrics.nan_detected)
        log_dict[f"{prefix}final_time_reached"] = int(metrics.final_time_reached)
    
    elif isinstance(metrics, MLMetrics):
        log_dict[f"{prefix}rare_regime_recall"] = metrics.rare_regime_recall
        log_dict[f"{prefix}rare_regime_precision"] = metrics.rare_regime_precision
        log_dict[f"{prefix}false_negative_rate"] = metrics.false_negative_rate
        log_dict[f"{prefix}calibration_error"] = metrics.calibration_error
        log_dict[f"{prefix}prediction_mse"] = metrics.prediction_mse
        log_dict[f"{prefix}prediction_mae"] = metrics.prediction_mae
    
    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)


def create_comparison_plots(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """
    Create visualization plots comparing integrators.
    
    Generates:
    1. Energy drift over time for each integrator
    2. Angular momentum drift over time
    3. Phase space trajectory projections
    
    Args:
        results: Output from evaluate_integrator_comparison
        save_path: Optional path to save figures
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed, skipping plots")
        return
    
    n_integrators = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_integrators))
    
    # Energy drift
    ax = axes[0, 0]
    for i, (name, data) in enumerate(results.items()):
        physics = data["physics"]
        ax.plot(physics.energy_drift.numpy(), label=name, color=colors[i])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Relative energy drift")
    ax.set_title("Energy Conservation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Angular momentum drift
    ax = axes[0, 1]
    for i, (name, data) in enumerate(results.items()):
        physics = data["physics"]
        ax.plot(physics.angular_momentum_drift.numpy(), label=name, color=colors[i])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Relative L drift")
    ax.set_title("Angular Momentum Conservation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log-scale energy drift
    ax = axes[1, 0]
    for i, (name, data) in enumerate(results.items()):
        physics = data["physics"]
        ax.semilogy(torch.abs(physics.energy_drift).numpy() + 1e-16, label=name, color=colors[i])
    ax.set_xlabel("Time step")
    ax.set_ylabel("|Relative energy drift|")
    ax.set_title("Energy Conservation (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary bar chart
    ax = axes[1, 1]
    names = list(results.keys())
    max_drifts = [results[n]["physics"].energy_drift_max for n in names]
    ax.bar(names, max_drifts, color=colors[:len(names)])
    ax.set_ylabel("Max energy drift")
    ax.set_title("Maximum Energy Drift Comparison")
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
