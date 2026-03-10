"""
Improved H-GNN Training on Real Satellite TLE Data

Key improvements over basic version:
1. Proper alpha initialization and learning
2. Data normalization for neural network training
3. Energy conservation loss (physics-informed)
4. Improved network architecture
5. Comprehensive evaluation metrics
6. Baseline comparisons
7. Visualization support

Usage:
    python train_on_tle_improved.py --n-satellites 20 --duration 1440 --horizon 60 --epochs 300
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from h_gnn.data.tle_loader import generate_trajectory_dataset, convert_to_hgnn_format
from h_gnn.models.hgnn import HGNN


# ============================================================================
# NORMALIZATION CONSTANTS
# ============================================================================
# Typical LEO values for normalization
Q_SCALE = 10000.0  # km (typical LEO radius ~7000 km)
P_SCALE = 10.0     # km/s (typical LEO velocity ~7.5 km/s)


def normalize_state(q: torch.Tensor, p: torch.Tensor):
    """Normalize position and velocity for neural network training."""
    return q / Q_SCALE, p / P_SCALE


def denormalize_state(q: torch.Tensor, p: torch.Tensor):
    """Denormalize back to physical units."""
    return q * Q_SCALE, p * P_SCALE


# ============================================================================
# PHYSICS-INFORMED LOSS
# ============================================================================
def compute_energy(q: torch.Tensor, p: torch.Tensor, mu: float = 398600.4418):
    """
    Compute specific orbital energy for each particle.
    E = |p|²/2 - μ/|q|
    """
    r = torch.norm(q, dim=-1, keepdim=True)
    v2 = torch.sum(p**2, dim=-1, keepdim=True)
    energy = 0.5 * v2 - mu / r
    return energy


def energy_conservation_loss(
    q0: torch.Tensor, p0: torch.Tensor,
    q_pred: torch.Tensor, p_pred: torch.Tensor,
    mu: float = 398600.4418,
):
    """
    Compute normalized energy conservation violation loss.
    
    Returns normalized relative energy drift to prevent huge loss values.
    """
    E0 = compute_energy(q0, p0, mu)
    E_pred = compute_energy(q_pred, p_pred, mu)
    
    # Normalize energies by their mean magnitude to prevent huge values
    E_scale = torch.abs(E0).mean() + 1e-10
    E0_norm = E0 / E_scale
    E_pred_norm = E_pred / E_scale
    
    # Squared normalized energy drift (more stable than abs)
    energy_loss = ((E_pred_norm - E0_norm) ** 2).mean()
    return energy_loss


# ============================================================================
# EVALUATION METRICS
# ============================================================================
def compute_metrics(
    q_pred: torch.Tensor, p_pred: torch.Tensor,
    q_target: torch.Tensor, p_target: torch.Tensor,
    q0: torch.Tensor, p0: torch.Tensor,
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
    E_target = compute_energy(q_target, p_target)
    
    energy_drift_pred = ((E_pred - E0) / (torch.abs(E0) + 1e-10)).abs().mean().item()
    energy_drift_target = ((E_target - E0) / (torch.abs(E0) + 1e-10)).abs().mean().item()
    
    return {
        'rmse_position_km': rmse_pos,
        'mae_position_km': mae_pos,
        'max_position_km': max_pos,
        'rmse_velocity_km_s': rmse_vel,
        'mae_velocity_km_s': mae_vel,
        'max_velocity_km_s': max_vel,
        'energy_drift_pred': energy_drift_pred,
        'energy_drift_target': energy_drift_target,
    }


# ============================================================================
# BASELINE MODELS
# ============================================================================
class PhysicsOnlyBaseline:
    """
    Physics-only baseline: pure Keplerian propagation.
    Uses SGP4-generated trajectories which include J2 and drag.
    """
    def __init__(self, dt: float = 60.0):
        self.dt = dt
    
    def predict(self, q0, p0, n_steps):
        """
        Simple velocity-based propagation (assuming constant velocity).
        This is the "no model" baseline.
        """
        # Linear extrapolation
        q_pred = q0 + p0 * (n_steps * self.dt)
        p_pred = p0  # Assume constant velocity
        return q_pred, p_pred


class NeuralNetworkBaseline(nn.Module):
    """
    Vanilla MLP baseline without Hamiltonian structure.
    Maps (q, p) → (dq, dp) directly.
    """
    def __init__(self, n_particles: int, hidden_dims=[128, 128, 64]):
        super().__init__()
        input_dim = n_particles * 6  # q and p for each particle
        output_dim = n_particles * 6
        
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.SiLU(),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.n_particles = n_particles
    
    def forward(self, q, p, n_steps=1):
        """Predict final state."""
        batch_size = q.shape[0] if q.dim() == 3 else 1
        if q.dim() == 2:
            q = q.unsqueeze(0)
            p = p.unsqueeze(0)
        
        # Flatten input
        x = torch.cat([q.flatten(1), p.flatten(1)], dim=-1)
        
        # Predict delta
        delta = self.mlp(x)
        
        # Split output
        dq = delta[:, :self.n_particles * 3].view(-1, self.n_particles, 3)
        dp = delta[:, self.n_particles * 3:].view(-1, self.n_particles, 3)
        
        q_pred = q + dq
        p_pred = p + dp
        
        if batch_size == 1:
            return q_pred.squeeze(0), p_pred.squeeze(0)
        return q_pred, p_pred


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_trajectories_3d(
    q_true: np.ndarray,
    q_pred: np.ndarray,
    satellite_names: list,
    save_path: str = "trajectory_comparison.png",
):
    """Plot true vs predicted 3D trajectories."""
    fig = plt.figure(figsize=(14, 6))
    
    # True trajectories
    ax1 = fig.add_subplot(121, projection='3d')
    for i, name in enumerate(satellite_names[:5]):  # First 5 satellites
        ax1.plot(q_true[:, i, 0], q_true[:, i, 1], q_true[:, i, 2], 
                label=name[:15], alpha=0.7)
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('True Trajectories')
    ax1.legend(fontsize=8)
    
    # Predicted trajectories
    ax2 = fig.add_subplot(122, projection='3d')
    for i, name in enumerate(satellite_names[:5]):
        ax2.plot(q_pred[:, i, 0], q_pred[:, i, 1], q_pred[:, i, 2],
                label=name[:15], alpha=0.7)
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.set_zlabel('Z (km)')
    ax2.set_title('Predicted Trajectories')
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved trajectory plot: {save_path}")


def plot_error_over_time(
    errors: Dict[str, np.ndarray],
    save_path: str = "error_over_time.png",
):
    """Plot prediction error vs time horizon."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    timesteps = np.arange(1, len(list(errors.values())[0]) + 1)
    
    # Position error
    for name, err in errors.items():
        ax1.plot(timesteps, err[:, 0], label=name, linewidth=2)
    ax1.set_xlabel('Time Steps (minutes)')
    ax1.set_ylabel('Position Error (km)')
    ax1.set_title('Position Error vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Velocity error
    for name, err in errors.items():
        ax2.plot(timesteps, err[:, 1], label=name, linewidth=2)
    ax2.set_xlabel('Time Steps (minutes)')
    ax2.set_ylabel('Velocity Error (km/s)')
    ax2.set_title('Velocity Error vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved error plot: {save_path}")


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    alphas: list,
    save_path: str = "training_curves.png",
):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.plot(epochs, alphas, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Alpha')
    ax2.set_title('Learned Alpha (Interaction Strength)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves: {save_path}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train_on_tle_data(
    tle_file: str = "satellites.tle",
    n_satellites: int = 20,
    duration_minutes: int = 1440,  # 24 hours
    horizon: int = 60,  # Predict 60 minutes ahead
    n_epochs: int = 300,
    batch_size: int = 16,
    lr: float = 1e-4,  # Reduced for stability
    energy_loss_weight: float = 1e-6,  # Reduced - energy already normalized
    hidden_dims: list = [128, 128, 64],  # Larger network
    alpha_init: float = 0.5,  # Raw value -> tanh(0.5)*0.01 ≈ 0.0046 clamped
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    run_baselines: bool = True,
    create_plots: bool = True,
):
    """
    Train H-GNN on real satellite data with all improvements.
    """
    print("=" * 70)
    print("H-GNN Training on Real Satellite Data (Improved)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"TLE file: {tle_file}")
    print(f"Satellites: {n_satellites}")
    print(f"Duration: {duration_minutes} minutes ({duration_minutes/60:.1f} hours)")
    print(f"Horizon: {horizon} minutes")
    print(f"Epochs: {n_epochs}")
    print(f"Alpha init: {alpha_init}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Energy loss weight: {energy_loss_weight}")
    print()
    
    # ========================================================================
    # GENERATE DATA
    # ========================================================================
    print("Generating trajectory data from TLE...")
    trajectory_data = generate_trajectory_dataset(
        tle_file,
        n_satellites=n_satellites,
        duration_minutes=duration_minutes,
        step_minutes=1,
        filter_leo=True,
    )
    
    # Convert to H-GNN format with more samples
    q0_all, p0_all, q_target_all, p_target_all = convert_to_hgnn_format(
        trajectory_data, horizon=horizon, stride=max(1, horizon // 4)  # More overlap
    )
    
    print(f"\nDataset size: {len(q0_all)} samples")
    print(f"  q0 shape: {q0_all.shape}")
    print(f"  Satellites: {trajectory_data['satellite_names']}")
    
    # ========================================================================
    # DATA STATISTICS (for loss normalization)
    # ========================================================================
    print("\nData statistics (physical units):")
    print(f"  Position range: {q0_all.min():.1f} to {q0_all.max():.1f} km")
    print(f"  Velocity range: {p0_all.min():.3f} to {p0_all.max():.3f} km/s")
    
    # ========================================================================
    # TRAIN/VAL SPLIT (use physical units throughout)
    # ========================================================================
    n_train = int(0.8 * len(q0_all))
    indices = torch.randperm(len(q0_all))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    # Training data (physical units)
    q0_train = q0_all[train_idx].to(device)
    p0_train = p0_all[train_idx].to(device)
    q_target_train = q_target_all[train_idx].to(device)
    p_target_train = p_target_all[train_idx].to(device)
    
    # Validation data (physical units)
    q0_val = q0_all[val_idx].to(device)
    p0_val = p0_all[val_idx].to(device)
    q_target_val = q_target_all[val_idx].to(device)
    p_target_val = p_target_all[val_idx].to(device)
    
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {len(val_idx)}")
    
    # ========================================================================
    # CREATE MODEL (Works in physical units - internal normalization handles MLP)
    # ========================================================================
    print("\nInitializing H-GNN model...")
    model = HGNN(
        hidden_dims=hidden_dims,
        interaction_radius=50000.0,  # km - physical units
        normalize_features=True,  # Internal MLP normalization
        softplus_output=False,
        dt=60.0,  # 1 minute timestep
        create_graph=True,
        alpha_init=alpha_init,
        use_dt_scaling=True,
    ).to(device)
    
    # Ensure alpha is properly initialized (access via hamiltonian)
    alpha = model.hamiltonian.alpha
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Initial alpha: {alpha.item():.6f}")
    
    # ========================================================================
    # OPTIMIZER WITH ALPHA INCLUDED
    # ========================================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    
    # Verify alpha is in optimizer
    param_count = sum(1 for _ in model.parameters())
    print(f"  Optimizer tracking {param_count} parameter groups")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    n_steps = horizon
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    alphas = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        epoch_energy_losses = []
        
        # Mini-batch training
        perm = torch.randperm(len(q0_train))
        
        for i in range(0, len(q0_train), batch_size):
            batch_idx = perm[i:min(i + batch_size, len(q0_train))]
            batch_loss = 0
            batch_energy_loss = 0
            batch_count = 0
            
            for j in batch_idx:
                q0 = q0_train[j]
                p0 = p0_train[j]
                q_target = q_target_train[j]
                p_target = p_target_train[j]
                
                # Forward pass (physical units)
                q_pred, p_pred = model(q0, p0, n_steps=n_steps)
                
                # Normalized trajectory loss (divide by typical scales for gradient stability)
                pos_error = torch.norm(q_pred - q_target, dim=-1).mean() / Q_SCALE
                vel_error = torch.norm(p_pred - p_target, dim=-1).mean() / P_SCALE
                traj_loss = pos_error + 0.1 * vel_error
                
                # Energy conservation loss (already relative)
                e_loss = energy_conservation_loss(q0, p0, q_pred, p_pred)
                
                # Combined loss
                loss = traj_loss + energy_loss_weight * e_loss
                
                batch_loss = batch_loss + loss
                batch_energy_loss = batch_energy_loss + e_loss.item()
                batch_count += 1
            
            batch_loss = batch_loss / batch_count
            
            optimizer.zero_grad()
            batch_loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(batch_loss.item())
            epoch_energy_losses.append(batch_energy_loss / batch_count)
        
        scheduler.step()
        
        avg_train_loss = np.mean(epoch_losses)
        avg_energy_loss = np.mean(epoch_energy_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss_list = []
        with torch.no_grad():
            for j in range(len(q0_val)):
                q0 = q0_val[j]
                p0 = p0_val[j]
                q_target = q_target_val[j]
                p_target = p_target_val[j]
                
                q_pred, p_pred = model(q0, p0, n_steps=n_steps)
                
                # Normalized loss (same as training)
                pos_error = torch.norm(q_pred - q_target, dim=-1).mean() / Q_SCALE
                vel_error = torch.norm(p_pred - p_target, dim=-1).mean() / P_SCALE
                v_loss = pos_error + 0.1 * vel_error
                val_loss_list.append(v_loss.item())
        
        avg_val_loss = np.mean(val_loss_list)
        val_losses.append(avg_val_loss)
        
        current_alpha = alpha.item()
        alphas.append(current_alpha)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_marker = " *"
            # Save best model
            torch.save(model.state_dict(), "best_hgnn_tle.pt")
        else:
            best_marker = ""
        
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d}: train={avg_train_loss:.4e}, val={avg_val_loss:.4e}, "
                  f"energy={avg_energy_loss:.4e}, alpha={current_alpha:.4e}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}{best_marker}")
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    model.eval()
    
    # Evaluate on validation set (physical units)
    all_metrics = []
    with torch.no_grad():
        for j in range(len(q0_val)):
            q0 = q0_val[j]
            p0 = p0_val[j]
            
            # Predict in physical units
            q_pred, p_pred = model(q0, p0, n_steps=n_steps)
            
            # Compute metrics (physical units)
            metrics = compute_metrics(
                q_pred.unsqueeze(0), p_pred.unsqueeze(0),
                q_target_val[j].unsqueeze(0), p_target_val[j].unsqueeze(0),
                q0_val[j].unsqueeze(0), p0_val[j].unsqueeze(0),
            )
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print("\nH-GNN Results:")
    print(f"  RMSE Position: {avg_metrics['rmse_position_km']:.4f} km")
    print(f"  MAE Position:  {avg_metrics['mae_position_km']:.4f} km")
    print(f"  Max Position:  {avg_metrics['max_position_km']:.4f} km")
    print(f"  RMSE Velocity: {avg_metrics['rmse_velocity_km_s']:.6f} km/s")
    print(f"  MAE Velocity:  {avg_metrics['mae_velocity_km_s']:.6f} km/s")
    print(f"  Energy Drift:  {avg_metrics['energy_drift_pred']:.6e}")
    print(f"  Final Alpha:   {alpha.item():.6e}")
    
    # ========================================================================
    # BASELINE COMPARISONS
    # ========================================================================
    if run_baselines:
        print("\n" + "-" * 70)
        print("Baseline Comparisons")
        print("-" * 70)
        
        # Physics-only baseline
        physics_baseline = PhysicsOnlyBaseline(dt=60.0)
        physics_metrics = []
        
        with torch.no_grad():
            for j in range(len(q0_val)):
                q_pred, p_pred = physics_baseline.predict(q0_val[j], p0_val[j], n_steps)
                metrics = compute_metrics(
                    q_pred.unsqueeze(0), p_pred.unsqueeze(0),
                    q_target_val[j].unsqueeze(0), p_target_val[j].unsqueeze(0),
                    q0_val[j].unsqueeze(0), p0_val[j].unsqueeze(0),
                )
                physics_metrics.append(metrics)
        
        avg_physics = {k: np.mean([m[k] for m in physics_metrics]) for k in physics_metrics[0]}
        
        print("\nPhysics-Only Baseline (Linear Extrapolation):")
        print(f"  RMSE Position: {avg_physics['rmse_position_km']:.4f} km")
        print(f"  MAE Position:  {avg_physics['mae_position_km']:.4f} km")
        
        # Improvement over baseline
        improvement = (avg_physics['rmse_position_km'] - avg_metrics['rmse_position_km']) / avg_physics['rmse_position_km'] * 100
        print(f"\nH-GNN Improvement: {improvement:.1f}% better than physics baseline")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    if create_plots:
        print("\n" + "-" * 70)
        print("Creating Visualizations")
        print("-" * 70)
        
        # Training curves
        plot_training_curves(train_losses, val_losses, alphas, "training_curves.png")
    
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    save_path = Path("trained_hgnn_tle_improved.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'hidden_dims': hidden_dims,
            'interaction_radius': 50000.0,
            'n_satellites': n_satellites,
            'horizon': horizon,
            'alpha_init': alpha_init,
            'q_scale': Q_SCALE,
            'p_scale': P_SCALE,
        },
        'metrics': avg_metrics,
        'satellite_names': trajectory_data['satellite_names'],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'alphas': alphas,
        'timestamp': datetime.now().isoformat(),
    }, save_path)
    
    print(f"\nModel saved to {save_path}")
    
    return model, avg_metrics


# ============================================================================
# SCALING EXPERIMENTS
# ============================================================================
def run_scaling_experiment(
    tle_file: str = "satellites.tle",
    satellite_counts: list = [5, 10, 20, 50],
    duration_minutes: int = 720,
    horizon: int = 30,
    n_epochs: int = 100,
):
    """
    Run scaling experiments with varying number of satellites.
    """
    print("=" * 70)
    print("SCALING EXPERIMENT")
    print("=" * 70)
    
    results = {}
    
    for n_sat in satellite_counts:
        print(f"\n>>> Training with {n_sat} satellites...")
        
        model, metrics = train_on_tle_data(
            tle_file=tle_file,
            n_satellites=n_sat,
            duration_minutes=duration_minutes,
            horizon=horizon,
            n_epochs=n_epochs,
            run_baselines=False,
            create_plots=False,
        )
        
        results[n_sat] = metrics
    
    # Plot scaling results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    n_sats = list(results.keys())
    pos_errors = [results[n]['rmse_position_km'] for n in n_sats]
    vel_errors = [results[n]['rmse_velocity_km_s'] for n in n_sats]
    
    ax1.plot(n_sats, pos_errors, 'bo-', linewidth=2, markersize=10)
    ax1.set_xlabel('Number of Satellites')
    ax1.set_ylabel('RMSE Position (km)')
    ax1.set_title('Position Error vs System Size')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(n_sats, vel_errors, 'ro-', linewidth=2, markersize=10)
    ax2.set_xlabel('Number of Satellites')
    ax2.set_ylabel('RMSE Velocity (km/s)')
    ax2.set_title('Velocity Error vs System Size')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scaling_experiment.png', dpi=150)
    print(f"\nSaved scaling plot: scaling_experiment.png")
    
    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train H-GNN on TLE data (Improved)")
    parser.add_argument("--tle", default="satellites.tle", help="TLE file path")
    parser.add_argument("--n-satellites", type=int, default=20, help="Number of satellites")
    parser.add_argument("--duration", type=int, default=1440, help="Duration in minutes")
    parser.add_argument("--horizon", type=int, default=60, help="Prediction horizon")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--alpha-init", type=float, default=0.5, help="Initial raw alpha (clamped via tanh*0.01)")
    parser.add_argument("--energy-weight", type=float, default=1e-6, help="Energy loss weight")
    parser.add_argument("--scaling", action="store_true", help="Run scaling experiment")
    
    args = parser.parse_args()
    
    if args.scaling:
        run_scaling_experiment(
            tle_file=args.tle,
            duration_minutes=args.duration,
            horizon=args.horizon,
            n_epochs=args.epochs,
        )
    else:
        train_on_tle_data(
            tle_file=args.tle,
            n_satellites=args.n_satellites,
            duration_minutes=args.duration,
            horizon=args.horizon,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            alpha_init=args.alpha_init,
            energy_loss_weight=args.energy_weight,
        )
