"""
Comprehensive H-GNN Experiment Pipeline

This script implements the full research experiment pipeline:
1. Dataset generation from TLE
2. Three model baselines (Physics, Neural, H-GNN)
3. Long-horizon prediction experiments
4. Scaling experiments (varying satellite count)
5. Noise robustness tests
6. Visualization generation
7. Comparison tables

Usage:
    python experiments/run_experiments.py --experiment all
    python experiments/run_experiments.py --experiment long_horizon
    python experiments/run_experiments.py --experiment scaling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from h_gnn.data.tle_loader import generate_trajectory_dataset, convert_to_hgnn_format
from h_gnn.models.hgnn import HGNN

# ============================================================================
# CONSTANTS
# ============================================================================
MU = 398600.4418  # Earth gravitational parameter (km³/s²)
R_EARTH = 6371.0  # Earth radius (km)
Q_SCALE = 10000.0
P_SCALE = 10.0

# ============================================================================
# DATA CLASSES FOR RESULTS
# ============================================================================
@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment run."""
    model_name: str
    n_satellites: int
    horizon_minutes: int
    rmse_position_km: float
    mae_position_km: float
    max_position_km: float
    rmse_velocity_km_s: float
    mae_velocity_km_s: float
    energy_drift: float
    training_time_s: float = 0.0
    n_parameters: int = 0


# ============================================================================
# BASELINE MODELS
# ============================================================================
class LinearExtrapolationBaseline:
    """
    Simple linear extrapolation baseline (weak baseline).
    
    Predicts: q_future = q0 + p0 * delta_t
    
    This is the baseline used in train_on_tle_improved.py for the 91% improvement claim.
    """
    def __init__(self, dt: float = 60.0):
        self.dt = dt  # seconds per step
        self.name = "Linear Extrapolation"
    
    def predict(self, q0: torch.Tensor, p0: torch.Tensor, n_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple linear extrapolation."""
        delta_t = n_steps * self.dt  # total time in seconds
        q_pred = q0 + p0 * delta_t  # velocity is in km/s, so q is in km
        p_pred = p0.clone()  # assume constant velocity
        return q_pred, p_pred


class PhysicsOnlyBaseline:
    """
    Physics-only baseline using two-body gravitational model.
    
    Hamiltonian: H = |p|²/2 - μ/|q|
    
    Uses velocity Verlet integration.
    """
    def __init__(self, mu: float = MU, dt: float = 60.0):
        self.mu = mu
        self.dt = dt
        self.name = "Physics-Only (Two-Body)"
    
    def compute_acceleration(self, q: torch.Tensor) -> torch.Tensor:
        """Compute gravitational acceleration: a = -μ * q / |q|³"""
        r = torch.norm(q, dim=-1, keepdim=True)
        r = torch.clamp(r, min=100.0)  # Prevent division by zero
        return -self.mu * q / (r ** 3)
    
    def predict(self, q0: torch.Tensor, p0: torch.Tensor, n_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate using velocity Verlet."""
        q = q0.clone()
        p = p0.clone()
        
        for _ in range(n_steps):
            # Half step in velocity
            a = self.compute_acceleration(q)
            p_half = p + 0.5 * self.dt * a
            
            # Full step in position
            q = q + self.dt * p_half
            
            # Half step in velocity with new position
            a_new = self.compute_acceleration(q)
            p = p_half + 0.5 * self.dt * a_new
        
        return q, p


class PureNeuralNetwork(nn.Module):
    """
    Pure neural network baseline (no physics structure).
    
    Maps (q, p) → (Δq, Δp) directly using MLP.
    """
    def __init__(self, n_particles: int, hidden_dims: List[int] = [256, 256, 128]):
        super().__init__()
        self.n_particles = n_particles
        self.name = "Pure Neural Network"
        
        input_dim = n_particles * 6
        output_dim = n_particles * 6
        
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize with small weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, q: torch.Tensor, p: torch.Tensor, n_steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state."""
        # Normalize inputs
        q_norm = q / Q_SCALE
        p_norm = p / P_SCALE
        
        # Flatten
        x = torch.cat([q_norm.flatten(), p_norm.flatten()])
        
        # Predict delta
        delta = self.network(x)
        
        # Split and reshape
        n = self.n_particles
        dq = delta[:n*3].view(n, 3) * Q_SCALE * 0.01  # Scale output
        dp = delta[n*3:].view(n, 3) * P_SCALE * 0.01
        
        return q + dq, p + dp


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def compute_orbital_features(q: torch.Tensor, p: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute orbital mechanics features.
    
    Returns:
        Dictionary of orbital features for each particle.
    """
    # Distances from Earth center
    r = torch.norm(q, dim=-1)
    
    # Velocities
    v = torch.norm(p, dim=-1)
    
    # Specific orbital energy: E = v²/2 - μ/r
    energy = 0.5 * v**2 - MU / r
    
    # Semi-major axis: a = -μ/(2E)
    a = -MU / (2 * energy + 1e-10)
    
    # Angular momentum: h = r × v
    h = torch.cross(q, p, dim=-1)
    h_mag = torch.norm(h, dim=-1)
    
    # Eccentricity vector
    e_vec = torch.cross(p, h, dim=-1) / MU - q / r.unsqueeze(-1)
    e = torch.norm(e_vec, dim=-1)
    
    # Inclination
    inclination = torch.arccos(torch.clamp(h[..., 2] / (h_mag + 1e-10), -1, 1))
    
    return {
        'radius': r,
        'velocity': v,
        'energy': energy,
        'semi_major_axis': a,
        'angular_momentum': h_mag,
        'eccentricity': e,
        'inclination': inclination,
    }


def compute_pairwise_features(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise orbital features between all satellite pairs.
    
    Features:
    - distance
    - distance²
    - relative velocity
    - dot(Δq, Δv)
    """
    n = q.shape[0]
    features = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dq = q[i] - q[j]
            dp = p[i] - p[j]
            
            dist = torch.norm(dq)
            rel_vel = torch.norm(dp)
            dot_qv = torch.dot(dq, dp)
            
            features.append([dist, dist**2, rel_vel, dot_qv])
    
    return torch.tensor(features) if features else torch.zeros(1, 4)


# ============================================================================
# METRICS COMPUTATION
# ============================================================================
def compute_energy(q: torch.Tensor, p: torch.Tensor, mu: float = MU) -> torch.Tensor:
    """Compute specific orbital energy."""
    r = torch.norm(q, dim=-1)
    v2 = torch.sum(p**2, dim=-1)
    return 0.5 * v2 - mu / r


def compute_metrics(
    q_pred: torch.Tensor,
    p_pred: torch.Tensor,
    q_target: torch.Tensor,
    p_target: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    
    # Position error
    pos_error = torch.norm(q_pred - q_target, dim=-1)
    rmse_pos = torch.sqrt((pos_error**2).mean()).item()
    mae_pos = pos_error.mean().item()
    max_pos = pos_error.max().item()
    
    # Velocity error
    vel_error = torch.norm(p_pred - p_target, dim=-1)
    rmse_vel = torch.sqrt((vel_error**2).mean()).item()
    mae_vel = vel_error.mean().item()
    max_vel = vel_error.max().item()
    
    # Energy conservation
    E0 = compute_energy(q0, p0)
    E_pred = compute_energy(q_pred, p_pred)
    energy_drift = ((E_pred - E0).abs() / (E0.abs() + 1e-10)).mean().item()
    
    return {
        'rmse_position_km': rmse_pos,
        'mae_position_km': mae_pos,
        'max_position_km': max_pos,
        'rmse_velocity_km_s': rmse_vel,
        'mae_velocity_km_s': mae_vel,
        'max_velocity_km_s': max_vel,
        'energy_drift': energy_drift,
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_3d_orbits(
    q_true: np.ndarray,
    q_physics: np.ndarray,
    q_hgnn: np.ndarray,
    satellite_names: List[str],
    save_path: str = "orbit_comparison_3d.png",
):
    """
    Plot 3D orbit comparison: True vs Physics vs H-GNN.
    """
    fig = plt.figure(figsize=(18, 6))
    
    n_sats = min(3, q_true.shape[1])  # Show up to 3 satellites
    
    for idx in range(n_sats):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # Earth sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = R_EARTH * np.outer(np.cos(u), np.sin(v))
        y = R_EARTH * np.outer(np.sin(u), np.sin(v))
        z = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.3, color='blue')
        
        # Orbits
        ax.plot(q_true[:, idx, 0], q_true[:, idx, 1], q_true[:, idx, 2],
                'g-', linewidth=2, label='True', alpha=0.8)
        ax.plot(q_physics[:, idx, 0], q_physics[:, idx, 1], q_physics[:, idx, 2],
                'r--', linewidth=2, label='Physics', alpha=0.8)
        ax.plot(q_hgnn[:, idx, 0], q_hgnn[:, idx, 1], q_hgnn[:, idx, 2],
                'b-', linewidth=2, label='H-GNN', alpha=0.8)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(f'{satellite_names[idx][:20]}')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_over_time(
    errors_dict: Dict[str, np.ndarray],
    save_path: str = "error_over_time.png",
):
    """
    Plot position error vs time for all models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'Physics-Only': 'red', 'Pure NN': 'orange', 'H-GNN': 'blue'}
    
    for name, errors in errors_dict.items():
        timesteps = np.arange(1, len(errors) + 1)
        color = colors.get(name, 'gray')
        
        ax1.plot(timesteps, errors[:, 0], label=name, color=color, linewidth=2)
        ax2.plot(timesteps, errors[:, 1], label=name, color=color, linewidth=2)
    
    ax1.set_xlabel('Time Step (minutes)')
    ax1.set_ylabel('Position Error (km)')
    ax1.set_title('Position Error vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Time Step (minutes)')
    ax2.set_ylabel('Velocity Error (km/s)')
    ax2.set_title('Velocity Error vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_energy_conservation(
    energy_dict: Dict[str, np.ndarray],
    save_path: str = "energy_conservation.png",
):
    """
    Plot energy drift over time for all models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'Physics-Only': 'red', 'Pure NN': 'orange', 'H-GNN': 'blue'}
    
    for name, energies in energy_dict.items():
        timesteps = np.arange(len(energies))
        E0 = energies[0]
        drift = np.abs(energies - E0) / (np.abs(E0) + 1e-10)
        
        color = colors.get(name, 'gray')
        ax.plot(timesteps, drift, label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Time Step (minutes)')
    ax.set_ylabel('Relative Energy Drift |ΔE/E₀|')
    ax.set_title('Energy Conservation Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_scaling_results(
    results: Dict[int, Dict[str, float]],
    save_path: str = "scaling_results.png",
):
    """
    Plot error vs number of satellites.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    n_sats = sorted(results.keys())
    
    for model_name in ['Physics-Only', 'H-GNN']:
        rmse = [results[n].get(f'{model_name}_rmse_pos', np.nan) for n in n_sats]
        vel = [results[n].get(f'{model_name}_rmse_vel', np.nan) for n in n_sats]
        
        color = 'red' if model_name == 'Physics-Only' else 'blue'
        ax1.plot(n_sats, rmse, 'o-', label=model_name, color=color, linewidth=2, markersize=8)
        ax2.plot(n_sats, vel, 'o-', label=model_name, color=color, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Satellites')
    ax1.set_ylabel('RMSE Position (km)')
    ax1.set_title('Position Error vs System Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Number of Satellites')
    ax2.set_ylabel('RMSE Velocity (km/s)')
    ax2.set_title('Velocity Error vs System Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_horizon_results(
    results: Dict[int, Dict[str, Dict[str, float]]],
    save_path: str = "horizon_results.png",
):
    """
    Plot error vs prediction horizon.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons = sorted(results.keys())
    
    colors = {'Linear': 'gray', 'Physics-Only': 'red', 'H-GNN': 'blue'}
    labels = {'Linear': 'Linear Extrapolation', 'Physics-Only': 'Two-Body Physics', 'H-GNN': 'H-GNN'}
    
    for model_name in ['Linear', 'Physics-Only', 'H-GNN']:
        if model_name in results[horizons[0]]:
            rmse = [results[h][model_name]['rmse_position_km'] for h in horizons]
            ax.plot(horizons, rmse, 'o-', label=labels[model_name], 
                    color=colors[model_name], linewidth=2, markersize=8)
    
    ax.set_xlabel('Prediction Horizon (minutes)')
    ax.set_ylabel('RMSE Position (km)')
    ax.set_title('Error Growth with Prediction Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================
def train_hgnn_model(
    q0_train: torch.Tensor,
    p0_train: torch.Tensor,
    q_target_train: torch.Tensor,
    p_target_train: torch.Tensor,
    n_steps: int,
    n_epochs: int = 100,
    lr: float = 1e-4,
    device: str = "cpu",
) -> HGNN:
    """Train H-GNN model."""
    
    model = HGNN(
        hidden_dims=[128, 128, 64],
        interaction_radius=50000.0,
        normalize_features=True,
        softplus_output=False,
        dt=60.0,  # 1 minute timestep matching data
        create_graph=True,
        alpha_init=0.5,
        use_dt_scaling=True,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for j in range(len(q0_train)):
            q0 = q0_train[j]
            p0 = p0_train[j]
            q_target = q_target_train[j]
            p_target = p_target_train[j]
            
            q_pred, p_pred = model(q0, p0, n_steps=n_steps)
            
            pos_error = torch.norm(q_pred - q_target, dim=-1).mean() / Q_SCALE
            vel_error = torch.norm(p_pred - p_target, dim=-1).mean() / P_SCALE
            loss = pos_error + 0.1 * vel_error
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss = {total_loss/len(q0_train):.4e}")
    
    return model


def run_long_horizon_experiment(
    tle_file: str,
    horizons: List[int] = [15, 30, 60, 120],
    n_satellites: int = 10,
    duration_minutes: int = 720,
    n_epochs: int = 100,
    device: str = "cpu",
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Run long-horizon prediction experiment.
    
    Tests prediction accuracy for varying time horizons.
    """
    print("\n" + "=" * 70)
    print("LONG-HORIZON PREDICTION EXPERIMENT")
    print("=" * 70)
    
    results = {}
    
    for horizon in horizons:
        print(f"\n--- Horizon: {horizon} minutes ---")
        
        # Generate data
        trajectory_data = generate_trajectory_dataset(
            tle_file,
            n_satellites=n_satellites,
            duration_minutes=duration_minutes,
            step_minutes=1,
            filter_leo=True,
        )
        
        q0_all, p0_all, q_target_all, p_target_all = convert_to_hgnn_format(
            trajectory_data, horizon=horizon, stride=max(1, horizon // 4)
        )
        
        # Split
        n_train = int(0.8 * len(q0_all))
        q0_train = q0_all[:n_train].to(device)
        p0_train = p0_all[:n_train].to(device)
        q_target_train = q_target_all[:n_train].to(device)
        p_target_train = p_target_all[:n_train].to(device)
        
        q0_val = q0_all[n_train:].to(device)
        p0_val = p0_all[n_train:].to(device)
        q_target_val = q_target_all[n_train:].to(device)
        p_target_val = p_target_all[n_train:].to(device)
        
        # Train H-GNN
        print("Training H-GNN...")
        model = train_hgnn_model(
            q0_train, p0_train, q_target_train, p_target_train,
            n_steps=horizon, n_epochs=n_epochs, device=device
        )
        
        # Evaluate
        model.eval()
        physics_baseline = PhysicsOnlyBaseline(dt=60.0)
        linear_baseline = LinearExtrapolationBaseline(dt=60.0)
        
        horizon_results = {'Linear': [], 'Physics-Only': [], 'H-GNN': []}
        
        with torch.no_grad():
            for j in range(len(q0_val)):
                q0 = q0_val[j]
                p0 = p0_val[j]
                q_target = q_target_val[j]
                p_target = p_target_val[j]
                
                # Linear extrapolation baseline (weak)
                q_lin, p_lin = linear_baseline.predict(q0, p0, horizon)
                metrics_lin = compute_metrics(q_lin, p_lin, q_target, p_target, q0, p0)
                horizon_results['Linear'].append(metrics_lin)
                
                # Physics baseline (strong)
                q_phys, p_phys = physics_baseline.predict(q0, p0, horizon)
                metrics_phys = compute_metrics(q_phys, p_phys, q_target, p_target, q0, p0)
                horizon_results['Physics-Only'].append(metrics_phys)
                
                # H-GNN
                q_hgnn, p_hgnn = model(q0, p0, n_steps=horizon)
                metrics_hgnn = compute_metrics(q_hgnn, p_hgnn, q_target, p_target, q0, p0)
                horizon_results['H-GNN'].append(metrics_hgnn)
        
        # Average metrics
        results[horizon] = {}
        for model_name in ['Linear', 'Physics-Only', 'H-GNN']:
            avg_metrics = {
                k: np.mean([m[k] for m in horizon_results[model_name]])
                for k in horizon_results[model_name][0].keys()
            }
            results[horizon][model_name] = avg_metrics
            print(f"  {model_name}: RMSE = {avg_metrics['rmse_position_km']:.2f} km")
    
    return results


def run_scaling_experiment(
    tle_file: str,
    satellite_counts: List[int] = [5, 10, 20, 50],
    horizon: int = 30,
    duration_minutes: int = 720,
    n_epochs: int = 100,
    device: str = "cpu",
) -> Dict[int, Dict[str, float]]:
    """
    Run scaling experiment with varying satellite counts.
    """
    print("\n" + "=" * 70)
    print("SCALING EXPERIMENT")
    print("=" * 70)
    
    results = {}
    
    for n_sat in satellite_counts:
        print(f"\n--- {n_sat} Satellites ---")
        
        try:
            # Generate data
            trajectory_data = generate_trajectory_dataset(
                tle_file,
                n_satellites=n_sat,
                duration_minutes=duration_minutes,
                step_minutes=1,
                filter_leo=True,
            )
            
            q0_all, p0_all, q_target_all, p_target_all = convert_to_hgnn_format(
                trajectory_data, horizon=horizon, stride=max(1, horizon // 4)
            )
            
            if len(q0_all) < 5:
                print(f"  Skipping: not enough samples")
                continue
            
            # Split
            n_train = int(0.8 * len(q0_all))
            q0_train = q0_all[:n_train].to(device)
            p0_train = p0_all[:n_train].to(device)
            q_target_train = q_target_all[:n_train].to(device)
            p_target_train = p_target_all[:n_train].to(device)
            
            q0_val = q0_all[n_train:].to(device)
            p0_val = p0_all[n_train:].to(device)
            q_target_val = q_target_all[n_train:].to(device)
            p_target_val = p_target_all[n_train:].to(device)
            
            # Train H-GNN
            print("Training H-GNN...")
            model = train_hgnn_model(
                q0_train, p0_train, q_target_train, p_target_train,
                n_steps=horizon, n_epochs=n_epochs, device=device
            )
            
            # Evaluate
            model.eval()
            physics_baseline = PhysicsOnlyBaseline(dt=60.0)
            
            hgnn_metrics = []
            physics_metrics = []
            
            with torch.no_grad():
                for j in range(len(q0_val)):
                    q0 = q0_val[j]
                    p0 = p0_val[j]
                    q_target = q_target_val[j]
                    p_target = p_target_val[j]
                    
                    # Physics
                    q_phys, p_phys = physics_baseline.predict(q0, p0, horizon)
                    physics_metrics.append(compute_metrics(q_phys, p_phys, q_target, p_target, q0, p0))
                    
                    # H-GNN
                    q_hgnn, p_hgnn = model(q0, p0, n_steps=horizon)
                    hgnn_metrics.append(compute_metrics(q_hgnn, p_hgnn, q_target, p_target, q0, p0))
            
            results[n_sat] = {
                'Physics-Only_rmse_pos': np.mean([m['rmse_position_km'] for m in physics_metrics]),
                'Physics-Only_rmse_vel': np.mean([m['rmse_velocity_km_s'] for m in physics_metrics]),
                'H-GNN_rmse_pos': np.mean([m['rmse_position_km'] for m in hgnn_metrics]),
                'H-GNN_rmse_vel': np.mean([m['rmse_velocity_km_s'] for m in hgnn_metrics]),
                'n_parameters': sum(p.numel() for p in model.parameters()),
            }
            
            print(f"  Physics RMSE: {results[n_sat]['Physics-Only_rmse_pos']:.2f} km")
            print(f"  H-GNN RMSE:   {results[n_sat]['H-GNN_rmse_pos']:.2f} km")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return results


def run_noise_robustness_experiment(
    tle_file: str,
    noise_levels: List[float] = [0.0, 0.01, 0.05, 0.1],
    n_satellites: int = 10,
    horizon: int = 30,
    duration_minutes: int = 480,
    n_epochs: int = 100,
    device: str = "cpu",
) -> Dict[float, Dict[str, float]]:
    """
    Test robustness to noise in initial conditions.
    """
    print("\n" + "=" * 70)
    print("NOISE ROBUSTNESS EXPERIMENT")
    print("=" * 70)
    
    # Generate base data
    trajectory_data = generate_trajectory_dataset(
        tle_file,
        n_satellites=n_satellites,
        duration_minutes=duration_minutes,
        step_minutes=1,
        filter_leo=True,
    )
    
    q0_all, p0_all, q_target_all, p_target_all = convert_to_hgnn_format(
        trajectory_data, horizon=horizon, stride=max(1, horizon // 4)
    )
    
    # Split
    n_train = int(0.8 * len(q0_all))
    q0_train = q0_all[:n_train].to(device)
    p0_train = p0_all[:n_train].to(device)
    q_target_train = q_target_all[:n_train].to(device)
    p_target_train = p_target_all[:n_train].to(device)
    
    # Train once on clean data
    print("Training H-GNN on clean data...")
    model = train_hgnn_model(
        q0_train, p0_train, q_target_train, p_target_train,
        n_steps=horizon, n_epochs=n_epochs, device=device
    )
    model.eval()
    
    physics_baseline = PhysicsOnlyBaseline(dt=60.0)
    
    results = {}
    
    for noise in noise_levels:
        print(f"\n--- Noise level: {noise} ---")
        
        hgnn_metrics = []
        physics_metrics = []
        
        with torch.no_grad():
            for j in range(n_train, len(q0_all)):
                q0 = q0_all[j].to(device)
                p0 = p0_all[j].to(device)
                q_target = q_target_all[j].to(device)
                p_target = p_target_all[j].to(device)
                
                # Add noise
                if noise > 0:
                    q0_noisy = q0 + noise * torch.randn_like(q0) * Q_SCALE
                    p0_noisy = p0 + noise * torch.randn_like(p0) * P_SCALE
                else:
                    q0_noisy = q0
                    p0_noisy = p0
                
                # Physics
                q_phys, p_phys = physics_baseline.predict(q0_noisy, p0_noisy, horizon)
                physics_metrics.append(compute_metrics(q_phys, p_phys, q_target, p_target, q0, p0))
                
                # H-GNN
                q_hgnn, p_hgnn = model(q0_noisy, p0_noisy, n_steps=horizon)
                hgnn_metrics.append(compute_metrics(q_hgnn, p_hgnn, q_target, p_target, q0, p0))
        
        results[noise] = {
            'Physics-Only_rmse': np.mean([m['rmse_position_km'] for m in physics_metrics]),
            'H-GNN_rmse': np.mean([m['rmse_position_km'] for m in hgnn_metrics]),
        }
        
        print(f"  Physics RMSE: {results[noise]['Physics-Only_rmse']:.2f} km")
        print(f"  H-GNN RMSE:   {results[noise]['H-GNN_rmse']:.2f} km")
    
    return results


def generate_comparison_table(results: Dict, output_file: str = "comparison_table.txt"):
    """Generate formatted comparison table."""
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("H-GNN EXPERIMENT RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        if 'long_horizon' in results:
            f.write("LONG-HORIZON PREDICTION RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Horizon':<10} {'Linear (km)':<15} {'Physics (km)':<15} {'H-GNN (km)':<15} {'vs Linear':<12} {'vs Physics':<12}\n")
            f.write("-" * 80 + "\n")
            
            for horizon, data in sorted(results['long_horizon'].items()):
                lin = data.get('Linear', {}).get('rmse_position_km', float('nan'))
                phys = data['Physics-Only']['rmse_position_km']
                hgnn = data['H-GNN']['rmse_position_km']
                vs_linear = (lin - hgnn) / lin * 100 if lin > 0 else float('nan')
                vs_physics = (phys - hgnn) / phys * 100
                f.write(f"{horizon:<10} {lin:<15.2f} {phys:<15.2f} {hgnn:<15.2f} {vs_linear:<12.1f}% {vs_physics:<12.1f}%\n")
            f.write("\n")
            f.write("Note: Linear = linear extrapolation (q + v*t), Physics = two-body gravitational dynamics\n")
            f.write("Positive improvement = H-GNN is better, Negative = H-GNN is worse\n\n")
        
        if 'scaling' in results:
            f.write("SCALING EXPERIMENT RESULTS\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'N Satellites':<15} {'Physics (km)':<15} {'H-GNN (km)':<15} {'Params':<10}\n")
            f.write("-" * 50 + "\n")
            
            for n_sat, data in sorted(results['scaling'].items()):
                phys = data['Physics-Only_rmse_pos']
                hgnn = data['H-GNN_rmse_pos']
                params = data.get('n_parameters', 'N/A')
                f.write(f"{n_sat:<15} {phys:<15.2f} {hgnn:<15.2f} {params:<10}\n")
            f.write("\n")
        
        if 'noise' in results:
            f.write("NOISE ROBUSTNESS RESULTS\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Noise Level':<15} {'Physics (km)':<15} {'H-GNN (km)':<15}\n")
            f.write("-" * 50 + "\n")
            
            for noise, data in sorted(results['noise'].items()):
                phys = data['Physics-Only_rmse']
                hgnn = data['H-GNN_rmse']
                f.write(f"{noise:<15.2f} {phys:<15.2f} {hgnn:<15.2f}\n")
    
    print(f"Saved: {output_file}")


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="H-GNN Experiment Pipeline")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["all", "long_horizon", "scaling", "noise", "visualization"],
                       help="Which experiment to run")
    parser.add_argument("--tle", type=str, default="satellites.tle", help="TLE file")
    parser.add_argument("--n-satellites", type=int, default=10)
    parser.add_argument("--duration", type=int, default=720)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    all_results = {}
    
    if args.experiment in ["all", "long_horizon"]:
        print("\n" + "=" * 70)
        print("Running Long-Horizon Experiment...")
        print("=" * 70)
        
        horizon_results = run_long_horizon_experiment(
            args.tle,
            horizons=[15, 30, 60],
            n_satellites=args.n_satellites,
            duration_minutes=args.duration,
            n_epochs=args.epochs,
            device=device,
        )
        all_results['long_horizon'] = horizon_results
        
        # Plot
        plot_horizon_results(horizon_results, str(output_dir / "horizon_results.png"))
    
    if args.experiment in ["all", "scaling"]:
        print("\n" + "=" * 70)
        print("Running Scaling Experiment...")
        print("=" * 70)
        
        scaling_results = run_scaling_experiment(
            args.tle,
            satellite_counts=[5, 10, 20],
            horizon=30,
            duration_minutes=args.duration,
            n_epochs=args.epochs,
            device=device,
        )
        all_results['scaling'] = scaling_results
        
        # Plot
        plot_scaling_results(scaling_results, str(output_dir / "scaling_results.png"))
    
    if args.experiment in ["all", "noise"]:
        print("\n" + "=" * 70)
        print("Running Noise Robustness Experiment...")
        print("=" * 70)
        
        noise_results = run_noise_robustness_experiment(
            args.tle,
            noise_levels=[0.0, 0.01, 0.05],
            n_satellites=args.n_satellites,
            horizon=30,
            duration_minutes=args.duration,
            n_epochs=args.epochs,
            device=device,
        )
        all_results['noise'] = noise_results
    
    # Generate comparison table
    generate_comparison_table(all_results, str(output_dir / "comparison_table.txt"))
    
    # Save raw results
    with open(output_dir / "results.json", 'w') as f:
        # Convert to serializable format
        serializable = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                serializable[k] = {
                    str(k2): v2 if not isinstance(v2, dict) else {
                        str(k3): float(v3) if isinstance(v3, (np.floating, float)) else v3
                        for k3, v3 in v2.items()
                    }
                    for k2, v2 in v.items()
                }
        json.dump(serializable, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
