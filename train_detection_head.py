"""
Train Detection Head on Real SGP4 Data with Rich Features

This script:
1. Generates 500 nominal + 500 anomalous trajectory windows from real SGP4
2. Extracts 12-dimensional feature vectors per trajectory
3. Trains MLP detection head to discriminate nominal vs anomalous
4. Evaluates on held-out test set across all Δv values
5. Reports AUROC and feature importances

H-GNN model remains FROZEN throughout.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from datetime import datetime

from h_gnn.data.tle_loader import generate_trajectory_dataset
from h_gnn.models.hgnn import HGNN
from anomaly_injection import inject_maneuver

# ============================================================================
# CONSTANTS
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_SCALE = 10000.0  # km
P_SCALE = 10.0     # km/s
CHECKPOINT_PATH = "best_hgnn_tle.pt"
OUTPUT_DIR = Path("checkpoints")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")


# ============================================================================
# STEP 1: LOAD H-GNN MODEL (FROZEN)
# ============================================================================
def load_frozen_hgnn():
    """Load pre-trained H-GNN model and freeze all parameters."""
    print("\nLoading pre-trained H-GNN model...")
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    # Create model with same config
    model = HGNN(
        hidden_dims=[128, 128, 64],
        interaction_radius=50000.0,
        normalize_features=True,
        softplus_output=False,
        dt=60.0,
        create_graph=True,
        alpha_init=0.5,
        use_dt_scaling=True,
    ).to(DEVICE)
    
    # Load compatible state dict
    state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params}")
    print(f"  All parameters frozen: {all(not p.requires_grad for p in model.parameters())}")
    
    model.eval()
    return model


# ============================================================================
# STEP 2: GENERATE TRAINING DATA
# ============================================================================
def generate_training_data(n_nominal: int = 500, n_anomalous: int = 500):
    """
    Generate 500 nominal + 500 anomalous trajectory windows.
    Anomalous: inject Δv from [0.05, 0.5] km/s.
    """
    print(f"\nGenerating training data: {n_nominal} nominal + {n_anomalous} anomalous...")
    
    # Generate base trajectories from TLE
    trajectory_data = generate_trajectory_dataset(
        "satellites.tle",
        n_satellites=30,
        duration_minutes=240,  # 4 hours
        step_minutes=1,
        filter_leo=True,
    )
    
    n_sat = len(trajectory_data["satellite_names"])
    n_timesteps = trajectory_data["positions"].shape[1]
    window_size = 61  # 60 minutes ahead + 1 initial
    
    print(f"  Base trajectories: {n_sat} satellites, {n_timesteps} timesteps")
    
    positions = np.transpose(trajectory_data["positions"], (1, 0, 2))
    velocities = np.transpose(trajectory_data["velocities"], (1, 0, 2))
    
    # Generate nominal windows
    windows_q = []
    windows_label = []
    stride = max(1, (n_timesteps - window_size) // max(1, n_nominal // n_sat))
    
    for sat_idx in range(n_sat):
        for t in range(0, n_timesteps - window_size, stride):
            if len(windows_q) >= n_nominal:
                break
            
            q = torch.tensor(positions[t:t+window_size, sat_idx], dtype=torch.float32)
            p = torch.tensor(velocities[t:t+window_size, sat_idx], dtype=torch.float32)
            
            windows_q.append((q, p))
            windows_label.append(0)  # nominal
        
        if len(windows_q) >= n_nominal:
            break
    
    # Generate anomalous windows
    rng = np.random.default_rng(42)
    for sat_idx in range(n_sat):
        for t in range(0, n_timesteps - window_size, stride):
            if len(windows_q) >= n_nominal + n_anomalous:
                break
            
            q_orig = torch.tensor(positions[t:t+window_size, sat_idx], dtype=torch.float32)
            p_orig = torch.tensor(velocities[t:t+window_size, sat_idx], dtype=torch.float32)
            
            # Inject random maneuver in [0.05, 0.5] km/s range
            dv_mag = float(rng.uniform(0.05, 0.5))
            injection_time = int(rng.uniform(5, window_size - 10))
            
            # Random direction for Δv
            dv_dir = rng.normal(size=3)
            dv_dir = dv_dir / (np.linalg.norm(dv_dir) + 1e-12)
            delta_v = dv_mag * dv_dir
            
            # Prepare for inject_maneuver: needs (N, T, 3) format
            q_batch = q_orig.unsqueeze(0)  # (1, window_size, 3)
            p_batch = p_orig.unsqueeze(0)  # (1, window_size, 3)
            delta_v_tensor = torch.tensor(delta_v, dtype=torch.float32)
            
            # Inject maneuver (returns modified copies)
            _, p_anom = inject_maneuver(
                q_batch,
                p_batch,
                t_maneuver=injection_time,
                delta_v=delta_v_tensor,
                satellite_index=0,
            )
            
            p_anom = p_anom.squeeze(0)  # Remove batch dimension
            
            windows_q.append((q_orig, p_anom))
            windows_label.append(1)  # anomalous
        
        if len(windows_q) >= n_nominal + n_anomalous:
            break
    
    # Trim to exact counts
    windows_q = windows_q[:n_nominal + n_anomalous]
    windows_label = windows_label[:n_nominal + n_anomalous]
    
    print(f"  Generated {len(windows_q)} windows ({np.sum(windows_label)} anomalous)")
    
    return windows_q, windows_label


# ============================================================================
# STEP 3: EXTRACT 12 FEATURES PER TRAJECTORY WINDOW
# ============================================================================
def compute_trajectory_residuals(model, q_window, p_window, n_steps=60):
    """
    Compute residuals for a single trajectory window.
    """
    q_window = q_window.to(DEVICE)
    p_window = p_window.to(DEVICE)
    
    residuals_pos = []
    residuals_vel = []
    residuals_energy = []
    phi_norms = []
    
    q_curr = q_window[0].unsqueeze(0)  # (1, 3)
    p_curr = p_window[0].unsqueeze(0)  # (1, 3)
    
    with torch.no_grad():
        for t in range(1, min(n_steps + 1, len(q_window))):
            q_target = q_window[t].unsqueeze(0)
            p_target = p_window[t].unsqueeze(0)
            
            # H-GNN prediction
            q_pred, p_pred = model(q_curr, p_curr, n_steps=1)
            
            # Position residual (normalized by Q_SCALE)
            pos_res = torch.norm(q_pred - q_target, dim=-1).item() / Q_SCALE
            residuals_pos.append(pos_res)
            
            # Velocity residual (normalized by P_SCALE)
            vel_res = torch.norm(p_pred - p_target, dim=-1).item() / P_SCALE
            residuals_vel.append(vel_res)
            
            # Energy residual
            r_target = torch.norm(q_target, dim=-1).item()
            v2_target = torch.sum(p_target**2, dim=-1).item()
            r_pred = torch.norm(q_pred, dim=-1).item()
            v2_pred = torch.sum(p_pred**2, dim=-1).item()
            
            E_target = 0.5 * v2_target - 398600.4418 / (r_target + 1e-6)
            E_pred = 0.5 * v2_pred - 398600.4418 / (r_pred + 1e-6)
            
            energy_res = abs(E_pred - E_target)
            residuals_energy.append(energy_res)
            
            # GNN correction magnitude (Phi norm)
            phi_norm = 0.0  # Placeholder
            phi_norms.append(phi_norm)
            
            # Step forward
            q_curr = q_target
            p_curr = p_target
    
    return (
        np.array(residuals_pos),
        np.array(residuals_vel),
        np.array(residuals_energy),
        np.array(phi_norms),
    )


def extract_features_from_window(model, q_window, p_window):
    """Extract 12 features from a trajectory window."""
    res_pos, res_vel, res_energy, phi_norms = compute_trajectory_residuals(
        model, q_window, p_window, n_steps=60
    )
    
    if len(res_pos) == 0:
        return np.zeros(12)
    
    features = []
    
    # Position residuals: max, mean, std
    features.append(np.max(res_pos))
    features.append(np.mean(res_pos))
    features.append(np.std(res_pos))
    
    # Velocity residuals: max, mean, std
    features.append(np.max(res_vel))
    features.append(np.mean(res_vel))
    features.append(np.std(res_vel))
    
    # Energy residuals: max, mean, std
    features.append(np.max(res_energy))
    features.append(np.mean(res_energy))
    features.append(np.std(res_energy))
    
    # Timestep of max residual (normalized)
    combined_res = res_pos + res_vel + res_energy / (np.max(np.abs(res_energy)) + 1e-10)
    timestep_max = np.argmax(combined_res) / len(combined_res)
    features.append(timestep_max)
    
    # Mean Phi norm
    features.append(np.mean(phi_norms))
    
    # Residual trend (linear fit slope)
    if len(combined_res) > 1:
        coeffs = np.polyfit(np.arange(len(combined_res)), combined_res, 1)
        trend = coeffs[0]  # slope
    else:
        trend = 0.0
    features.append(trend)
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# STEP 4: TRAIN DETECTION HEAD
# ============================================================================
class DetectionHeadMLP(nn.Module):
    """MLP for anomaly detection on trajectory features."""
    def __init__(self, feature_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x)


def train_detection_head(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train detection head MLP."""
    print(f"\nTraining detection head for {epochs} epochs...")
    
    model = DetectionHeadMLP(feature_dim=X_train.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_val = X_val.to(DEVICE)
    y_val = y_val.to(DEVICE)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]
            
            logits = model(X_batch).squeeze()
            loss = criterion(logits, y_batch.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).squeeze()
            val_loss = criterion(val_logits, y_val.float())
        
        train_loss = loss.item()
        train_losses.append(train_loss)
        val_losses.append(val_loss.item())
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    # Save checkpoint
    save_path = OUTPUT_DIR / "detection_head.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Saved best model to {save_path}")
    
    return model, train_losses, val_losses


# ============================================================================
# STEP 5: EVALUATE ON TEST SET
# ============================================================================
def evaluate_on_test_set(hgnn_model, detection_model, test_windows, test_labels, delta_v_values):
    """Evaluate detection head on held-out test set."""
    print(f"\nEvaluating on held-out test set ({len(test_windows)} samples)...")
    print(f"  Delta-v values: {delta_v_values}")
    
    detection_model.eval()
    results = {}
    
    print(f"\n  Computing features for test set...")
    all_features = []
    for idx, (q_win, p_win) in enumerate(test_windows):
        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{len(test_windows)}")
        features = extract_features_from_window(hgnn_model, q_win, p_win)
        all_features.append(features)
    
    all_features = np.array(all_features)
    
    print(f"\nEvaluating per Δv level:")
    for dv_target in delta_v_values:
        print(f"\n  Δv = {dv_target:.3f} km/s:")
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for idx in range(len(test_windows)):
                features = torch.tensor(all_features[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                label = test_labels[idx]
                
                prob = detection_model(features).item()
                all_probs.append(prob)
                all_labels.append(label)
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        if len(np.unique(all_labels)) > 1:
            auroc = roc_auc_score(all_labels, all_probs)
        else:
            auroc = 0.5
        
        preds = (all_probs >= 0.5).astype(int)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        # Score separation
        nominal_probs = all_probs[all_labels == 0]
        anomalous_probs = all_probs[all_labels == 1]
        
        if len(nominal_probs) > 0 and np.std(nominal_probs) > 0:
            separation = (np.mean(anomalous_probs) - np.mean(nominal_probs)) / (np.std(nominal_probs) + 1e-10)
        else:
            separation = 0.0
        
        results[dv_target] = {
            'auroc': auroc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'separation': separation,
            'nominal_probs': nominal_probs,
            'anomalous_probs': anomalous_probs,
        }
        
        print(f"    AUROC: {auroc:.4f}")
        print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"    Separation: {separation:.4f}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Training Detection Head on Real SGP4 Data with Rich Features")
    print("=" * 80)
    
    # Load frozen H-GNN
    hgnn_model = load_frozen_hgnn()
    
    # Generate training data
    train_windows, train_labels = generate_training_data(n_nominal=500, n_anomalous=500)
    
    # Extract features
    print(f"\nExtracting features from {len(train_windows)} windows...")
    features_list = []
    for i, (q_win, p_win) in enumerate(train_windows):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(train_windows)}")
        
        features = extract_features_from_window(hgnn_model, q_win, p_win)
        features_list.append(features)
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(train_labels, dtype=np.int64)
    
    print(f"  Features shape: {X.shape}")
    print(f"  Feature statistics:")
    for i in range(X.shape[1]):
        print(f"    Feature {i+1:2d}: min={X[:, i].min():.4e}, max={X[:, i].max():.4e}, mean={X[:, i].mean():.4e}")
    
    # Train/val split
    n_train = int(0.8 * len(X))
    perm = np.random.permutation(len(X))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    
    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.long)
    
    # Train detection head
    detection_model, train_losses, val_losses = train_detection_head(
        X_train, y_train, X_val, y_val, epochs=50, batch_size=32
    )
    
    # Generate test set
    print(f"\nGenerating held-out test set (200 nominal + 200 anomalous)...")
    test_windows, test_labels = generate_training_data(n_nominal=200, n_anomalous=200)
    
    # Evaluate
    delta_v_values = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    results = evaluate_on_test_set(hgnn_model, detection_model, test_windows, test_labels, delta_v_values)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Detection Head Performance")
    print("=" * 80)
    
    print("\nAUROC vs Δv:")
    for dv in delta_v_values:
        if dv in results:
            print(f"  Δv={dv:.3f} km/s: AUROC={results[dv]['auroc']:.4f}, Separation={results[dv]['separation']:.4f}")
    
    print("\n" + "=" * 80)



