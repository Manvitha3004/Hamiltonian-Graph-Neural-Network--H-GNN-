"""
Train H-GNN on Real Satellite TLE Data

This script demonstrates training the Hamiltonian Graph Neural Network
on real satellite trajectories derived from CelesTrak TLE data.

The model learns to predict gravitational corrections that improve
trajectory prediction beyond the base Keplerian approximation.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

from h_gnn.data.tle_loader import TLEDataset, generate_trajectory_dataset, convert_to_hgnn_format
from h_gnn.models.hgnn import HGNN


def train_on_tle_data(
    tle_file: str = "satellites.tle",
    n_satellites: int = 10,
    duration_minutes: int = 360,  # 6 hours
    horizon: int = 30,  # Predict 30 minutes ahead
    n_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train H-GNN on real satellite data.
    
    Args:
        tle_file: Path to TLE file
        n_satellites: Number of satellites to use
        duration_minutes: Length of trajectory to generate
        horizon: Prediction horizon in minutes
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
    """
    print("=" * 60)
    print("H-GNN Training on Real Satellite Data")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"TLE file: {tle_file}")
    print(f"Satellites: {n_satellites}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Horizon: {horizon} minutes")
    print()
    
    # Generate trajectory data
    print("Generating trajectory data from TLE...")
    trajectory_data = generate_trajectory_dataset(
        tle_file,
        n_satellites=n_satellites,
        duration_minutes=duration_minutes,
        step_minutes=1,
        filter_leo=True,
    )
    
    # Convert to H-GNN format
    q0_all, p0_all, q_target_all, p_target_all = convert_to_hgnn_format(
        trajectory_data, horizon=horizon, stride=horizon // 2
    )
    
    print(f"\nDataset size: {len(q0_all)} samples")
    print(f"  q0 shape: {q0_all.shape}")
    print(f"  Satellites: {trajectory_data['satellite_names']}")
    
    # Train/val split
    n_train = int(0.8 * len(q0_all))
    indices = torch.randperm(len(q0_all))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    q0_train = q0_all[train_idx].to(device)
    p0_train = p0_all[train_idx].to(device)
    q_target_train = q_target_all[train_idx].to(device)
    p_target_train = p_target_all[train_idx].to(device)
    
    q0_val = q0_all[val_idx].to(device)
    p0_val = p0_all[val_idx].to(device)
    q_target_val = q_target_all[val_idx].to(device)
    p_target_val = p_target_all[val_idx].to(device)
    
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {len(val_idx)}")
    
    # Create model
    print("\nInitializing H-GNN model...")
    model = HGNN(
        hidden_dims=[64, 64, 32],
        interaction_radius=50000.0,  # km - covers LEO satellites
        normalize_features=True,
        softplus_output=False,
        dt=60.0,  # 1 minute timestep
        create_graph=True,  # Enable gradients
        alpha_init=1e-3,  # Initialize alpha for orbital dynamics
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    alpha = getattr(model.interaction, 'alpha', None)
    if alpha is not None:
        print(f"  Initial alpha: {alpha.item():.6f}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    n_steps = horizon  # Number of integration steps
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        
        # Mini-batch training (process each sample individually)
        for i in range(0, len(q0_train), batch_size):
            batch_loss = 0
            batch_count = 0
            
            for j in range(i, min(i + batch_size, len(q0_train))):
                # Get single system
                q0 = q0_train[j]  # (n_particles, 3)
                p0 = p0_train[j]  # (n_particles, 3)
                q_target = q_target_train[j]
                p_target = p_target_train[j]
                
                # Forward pass: integrate for n_steps
                q_pred, p_pred = model(q0, p0, n_steps=n_steps)
                
                # Compute loss
                pos_error = torch.norm(q_pred - q_target, dim=-1).mean()
                vel_error = torch.norm(p_pred - p_target, dim=-1).mean()
                loss = pos_error + 0.1 * vel_error
                
                batch_loss = batch_loss + loss
                batch_count += 1
            
            # Average loss over batch
            batch_loss = batch_loss / batch_count
            
            optimizer.zero_grad()
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(batch_loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for j in range(len(q0_val)):
                q0 = q0_val[j]
                p0 = p0_val[j]
                q_target = q_target_val[j]
                p_target = p_target_val[j]
                
                q_pred, p_pred = model(q0, p0, n_steps=n_steps)
                
                pos_error_val = torch.norm(q_pred - q_target, dim=-1).mean()
                vel_error_val = torch.norm(p_pred - p_target, dim=-1).mean()
                val_loss = pos_error_val + 0.1 * vel_error_val
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_marker = " *"
        else:
            best_marker = ""
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            alpha_val = alpha.item() if alpha is not None else 0.0
            print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.4e}, "
                  f"val_loss={avg_val_loss:.4e}, alpha={alpha_val:.4e}"
                  f"{best_marker}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    model.eval()
    pos_errors = []
    vel_errors = []
    with torch.no_grad():
        for j in range(len(q0_val)):
            q0 = q0_val[j]
            p0 = p0_val[j]
            q_target = q_target_val[j]
            p_target = p_target_val[j]
            
            q_pred, p_pred = model(q0, p0, n_steps=n_steps)
            
            pos_errors.append(torch.norm(q_pred - q_target, dim=-1).mean().item())
            vel_errors.append(torch.norm(p_pred - p_target, dim=-1).mean().item())
    
    final_pos_error = np.mean(pos_errors)
    final_vel_error = np.mean(vel_errors)
    
    print(f"Position error: {final_pos_error:.4f} km")
    print(f"Velocity error: {final_vel_error:.6f} km/s")
    if alpha is not None:
        print(f"Final alpha: {alpha.item():.6e}")
    
    # Save model
    save_path = Path("trained_hgnn_tle.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'hidden_dims': [64, 64, 32],
            'interaction_radius': 50000.0,
            'n_satellites': n_satellites,
            'horizon': horizon,
        },
        'metrics': {
            'final_pos_error': final_pos_error,
            'final_vel_error': final_vel_error,
            'best_val_loss': float(best_val_loss),
        },
        'satellite_names': trajectory_data['satellite_names'],
        'timestamp': datetime.now().isoformat(),
    }, save_path)
    
    print(f"\nModel saved to {save_path}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train H-GNN on TLE data")
    parser.add_argument("--tle", default="satellites.tle", help="TLE file path")
    parser.add_argument("--n-satellites", type=int, default=10, help="Number of satellites")
    parser.add_argument("--duration", type=int, default=360, help="Duration in minutes")
    parser.add_argument("--horizon", type=int, default=30, help="Prediction horizon")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    args = parser.parse_args()
    
    train_on_tle_data(
        tle_file=args.tle,
        n_satellites=args.n_satellites,
        duration_minutes=args.duration,
        horizon=args.horizon,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
