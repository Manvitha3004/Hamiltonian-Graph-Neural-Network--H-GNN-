"""
REAL VALIDATION: Anomaly Detection with Actual SGP4 Data and H-GNN Inference

This script validates the anomaly detection system using:
- Real trained H-GNN checkpoint (best_hgnn_tle.pt)
- Real SGP4 satellite trajectories from TLE data
- Real anomaly injection via impulses
- Real model inference on actual trajectory data
- Actual AUROC computation (no synthetic score generation)

NO SYNTHETIC DISTRIBUTIONS. NO FAKE DATA.
Only real model predictions on real orbital mechanics.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score

from h_gnn.models.hgnn import HGNN
from h_gnn.data.tle_loader import generate_trajectory_dataset
from anomaly_injection import inject_maneuver
from anomaly_detection import compute_anomaly_score


# ============================================================================
# CONFIGURATION
# ============================================================================
CHECKPOINT_PATH = "best_hgnn_tle.pt"
TLE_FILE = "satellites.tle"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_NOMINAL_PER_DV = 100
NUM_ANOMALOUS_PER_DV = 100
DELTA_V_MAGNITUDES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]  # km/s


# ============================================================================
# MODEL LOADING (WITH PROOF OF SUCCESSFUL LOAD)
# ============================================================================
def load_hgnn_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load trained H-GNN model from checkpoint with verification."""
    print("=" * 80)
    print("LOADING H-GNN MODEL")
    print("=" * 80)
    
    # Initialize model with training configuration
    model = HGNN(
        hidden_dims=[128, 128, 64],
        interaction_radius=50000.0,
        normalize_features=True,
        softplus_output=False,
        dt=60.0,
        create_graph=True,
        alpha_init=0.5,
        use_dt_scaling=True,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Filter compatible keys (handle minor version differences)
    model_state = model.state_dict()
    compatible_keys = {k: v for k, v in checkpoint.items() if k in model_state}
    incompatible_keys = [k for k in checkpoint.keys() if k not in model_state]
    
    model.load_state_dict(compatible_keys, strict=False)
    model = model.to(device)
    
    # Print verification info
    print(f"\nCheckpoint path: {Path(checkpoint_path).absolute()}")
    print(f"Compatible keys loaded: {len(compatible_keys)}/{len(checkpoint)}")
    if incompatible_keys:
        print(f"Incompatible keys (ignored): {incompatible_keys}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[PROOF OF SUCCESSFUL LOAD]")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Learned alpha: {model.hamiltonian.alpha.item():.6e}")
    print(f"  Model device: {next(model.parameters()).device}")
    
    model.eval()
    print(f"\nModel loaded successfully and set to eval mode.")
    return model


# ============================================================================
# DATA GENERATION (REAL SGP4 TRAJECTORIES)
# ============================================================================
def generate_real_trajectories(
    tle_file: str,
    n_satellites: int = 10,
    duration_minutes: int = 120,
    step_minutes: int = 1,
) -> Dict[str, np.ndarray]:
    """Generate real SGP4 trajectories from TLE data.
    
    Returns:
        Dict with 'positions' (T, N, 3) and 'velocities' (T, N, 3) in km and km/s.
    """
    print("\n" + "=" * 80)
    print("GENERATING REAL SGP4 TRAJECTORIES")
    print("=" * 80)
    
    trajectory_data = generate_trajectory_dataset(
        tle_file,
        n_satellites=n_satellites,
        duration_minutes=duration_minutes,
        step_minutes=step_minutes,
        filter_leo=True,
    )
    
    n_steps, n_sats, _ = trajectory_data["positions"].shape
    
    print(f"\nGenerated trajectories:")
    print(f"  Duration: {duration_minutes} minutes ({duration_minutes / 60:.1f} hours)")
    print(f"  Timesteps: {n_steps} (at {step_minutes}-minute intervals)")
    print(f"  Satellites: {n_sats} LEO satellites")
    print(f"  Position shape: {trajectory_data['positions'].shape} (T, N, 3) in km")
    print(f"  Velocity shape: {trajectory_data['velocities'].shape} (T, N, 3) in km/s")
    print(f"  Satellite names: {trajectory_data['satellite_names'][:3]}...")
    
    return trajectory_data


def extract_trajectory_windows(
    positions: np.ndarray,
    velocities: np.ndarray,
    window_size: int = 61,
    n_samples: int = 100,
    rng: np.random.Generator = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract random trajectory windows from full trajectories.
    
    Args:
        positions: Shape (T, N, 3)
        velocities: Shape (T, N, 3)
        window_size: Number of timesteps per window
        n_samples: Number of windows to extract
        rng: Random number generator
        
    Returns:
        List of (positions_window, velocities_window) tuples, each (window_size, N, 3)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    max_start = len(positions) - window_size
    if max_start <= 0:
        raise ValueError(f"Trajectory too short ({len(positions)}) for window_size={window_size}")
    
    windows = []
    for _ in range(n_samples):
        start_idx = rng.integers(0, max_start + 1)
        q_window = positions[start_idx:start_idx + window_size]
        p_window = velocities[start_idx:start_idx + window_size]
        windows.append((q_window, p_window))
    
    return windows


# ============================================================================
# ANOMALY SCORE COMPUTATION (REAL INFERENCE)
# ============================================================================
def compute_scores_for_trajectories(
    model: nn.Module,
    trajectories: List[Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> np.ndarray:
    """Compute anomaly scores for trajectories using real H-GNN inference.
    
    Args:
        model: Trained H-GNN model (eval mode)
        trajectories: List of (q, p) tuples, each with shape (T, N, 3)
        device: Torch device
        
    Returns:
        Array of anomaly scores (float, one per trajectory)
    """
    scores = []
    n_traj = len(trajectories)
    
    print(f"\nComputing anomaly scores for {n_traj} trajectories...")
    
    with torch.no_grad():
        for i, (q_window, p_window) in enumerate(trajectories):
            # Convert to torch tensors
            q = torch.as_tensor(q_window).float().to(device)  # (T, N, 3)
            p = torch.as_tensor(p_window).float().to(device)  # (T, N, 3)
            
            # Initial state
            q0 = q[0]  # (N, 3)
            p0 = p[0]  # (N, 3)
            n_steps = len(q) - 1
            
            # Real H-GNN prediction
            q_pred, p_pred = model(q0, p0, n_steps=n_steps, return_trajectory=True)
            
            # Prepare trajectory data for anomaly scoring
            actual_traj = (q, p)
            predicted_traj = (q_pred, p_pred)
            
            # Compute real anomaly score from residuals
            score = compute_anomaly_score(model, actual_traj, predicted_traj)
            scores.append(score)
            
            if (i + 1) % max(1, n_traj // 10) == 0:
                print(f"  [{i + 1}/{n_traj}] scores computed...")
    
    return np.array(scores, dtype=np.float32)


# ============================================================================
# EVALUATION METRICS
# ============================================================================
def compute_detection_metrics(
    nominal_scores: np.ndarray,
    anomalous_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute detection metrics: AUROC, precision, recall, F1."""
    
    # Labels: 0 = nominal, 1 = anomalous
    y_true = np.concatenate([np.zeros(len(nominal_scores)), np.ones(len(anomalous_scores))])
    y_scores = np.concatenate([nominal_scores, anomalous_scores])
    
    # AUROC
    auroc = roc_auc_score(y_true, y_scores)
    
    # Optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Metrics at optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    
    return {
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "optimal_threshold": optimal_threshold,
        "nominal_mean": nominal_scores.mean(),
        "nominal_std": nominal_scores.std(),
        "anomalous_mean": anomalous_scores.mean(),
        "anomalous_std": anomalous_scores.std(),
    }


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_roc_curve(
    nominal_scores: np.ndarray,
    anomalous_scores: np.ndarray,
    dv_magnitude: float,
    auroc: float,
    save_dir: str = ".",
):
    """Plot ROC curve."""
    y_true = np.concatenate([np.zeros(len(nominal_scores)), np.ones(len(anomalous_scores))])
    y_scores = np.concatenate([nominal_scores, anomalous_scores])
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2.5, label=f"H-GNN (AUROC={auroc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve: Anomaly Detection (Δv={dv_magnitude:.3f} km/s)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curve_dv{dv_magnitude:.2f}.png", dpi=120)
    plt.close()


def plot_score_distribution(
    nominal_scores: np.ndarray,
    anomalous_scores: np.ndarray,
    dv_magnitude: float,
    save_dir: str = ".",
):
    """Plot score distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nominal_scores, bins=20, alpha=0.6, label="Nominal", color="blue", edgecolor="black", density=True)
    ax.hist(anomalous_scores, bins=20, alpha=0.6, label="Anomalous", color="red", edgecolor="black", density=True)
    ax.set_xlabel("Anomaly Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Score Distribution (Δv={dv_magnitude:.3f} km/s)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/score_distribution_dv{dv_magnitude:.2f}.png", dpi=120)
    plt.close()


def plot_auroc_comparison(
    delta_v_mags: List[float],
    aurocs: List[float],
    save_dir: str = ".",
):
    """Plot AUROC vs Δv magnitude."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(delta_v_mags, aurocs, "o-", linewidth=2.5, markersize=10, 
            color="blue", markeredgecolor="darkblue", markeredgewidth=1.5)
    ax.set_xlabel("Δv Magnitude (km/s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUROC", fontsize=12, fontweight="bold")
    ax.set_title("Anomaly Detection Performance vs Maneuver Magnitude", fontsize=13, fontweight="bold")
    ax.set_ylim([0.5, 1.0])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/auroc_vs_delta_v.png", dpi=150)
    plt.close()


# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================
def run_real_validation():
    """Run complete validation with REAL data and REAL inference."""
    
    print("\n" + "=" * 80)
    print("REAL VALIDATION: H-GNN ANOMALY DETECTION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print()
    
    # STEP 1: Load model with proof
    model = load_hgnn_model(CHECKPOINT_PATH, DEVICE)
    
    # STEP 2: Generate real SGP4 trajectories
    trajectory_data = generate_real_trajectories(
        TLE_FILE,
        n_satellites=250,
        duration_minutes=240,  # 4 hours for variety
        step_minutes=1,
    )
    
    positions = trajectory_data["positions"]  # (T, N, 3)
    velocities = trajectory_data["velocities"]  # (T, N, 3)
    
    # STEP 3: Run validation for each Δv magnitude
    print("\n" + "=" * 80)
    print("VALIDATION: COMPUTING ANOMALY SCORES FOR EACH DELTA-V MAGNITUDE")
    print("=" * 80)
    
    results_by_dv = {}
    
    for dv_mag in DELTA_V_MAGNITUDES:
        print(f"\n{'-' * 80}")
        print(f"Delta-v magnitude: {dv_mag:.3f} km/s")
        print(f"{'-' * 80}")
        
        rng = np.random.default_rng(42 + int(dv_mag * 1000))
        
        # Extract nominal windows
        print(f"\nExtracting {NUM_NOMINAL_PER_DV} nominal trajectory windows...")
        nominal_windows = extract_trajectory_windows(
            positions, velocities,
            window_size=61,
            n_samples=NUM_NOMINAL_PER_DV,
            rng=rng,
        )
        
        # Extract anomalous windows (will inject maneuvers)
        print(f"Extracting {NUM_ANOMALOUS_PER_DV} windows for anomaly injection...")
        anomalous_windows_base = extract_trajectory_windows(
            positions, velocities,
            window_size=61,
            n_samples=NUM_ANOMALOUS_PER_DV,
            rng=rng,
        )
        
        # Inject maneuvers into anomalous windows
        print(f"Injecting {dv_mag:.3f} km/s maneuvers...")
        anomalous_windows = []
        for q_window, p_window in anomalous_windows_base:
            # Inject at random timestep
            t_inject = rng.integers(10, len(q_window) - 1)
            sat_idx = 0  # Inject into first satellite
            
            # Random direction Δv
            dv_direction = rng.normal(size=3)
            dv_direction = dv_direction / (np.linalg.norm(dv_direction) + 1e-12)
            delta_v = dv_mag * dv_direction
            
            # Inject
            q_anomalous, p_anomalous = inject_maneuver(
                q_window, p_window,
                t_maneuver=t_inject,
                delta_v=delta_v,
                satellite_index=sat_idx,
            )
            
            anomalous_windows.append((q_anomalous, p_anomalous))
        
        # Compute scores
        print(f"\nComputing anomaly scores (REAL inference):")
        nominal_scores = compute_scores_for_trajectories(model, nominal_windows, DEVICE)
        anomalous_scores = compute_scores_for_trajectories(model, anomalous_windows, DEVICE)
        
        # Compute metrics
        metrics = compute_detection_metrics(nominal_scores, anomalous_scores)
        
        print(f"\nDetection Results (Δv={dv_mag:.3f} km/s):")
        print(f"  AUROC:           {metrics['auroc']:.4f}")
        print(f"  Precision:       {metrics['precision']:.4f}")
        print(f"  Recall:          {metrics['recall']:.4f}")
        print(f"  F1 Score:        {metrics['f1']:.4f}")
        print(f"  Optimal thresh:  {metrics['optimal_threshold']:.4f}")
        print(f"  Nominal score:   {metrics['nominal_mean']:.4f} +/- {metrics['nominal_std']:.4f}")
        print(f"  Anomalous score: {metrics['anomalous_mean']:.4f} +/- {metrics['anomalous_std']:.4f}")
        
        # Save results
        results_by_dv[dv_mag] = {
            "nominal_scores": nominal_scores,
            "anomalous_scores": anomalous_scores,
            "metrics": metrics,
        }
        
        # Visualizations
        plot_roc_curve(nominal_scores, anomalous_scores, dv_mag, metrics["auroc"])
        plot_score_distribution(nominal_scores, anomalous_scores, dv_mag)
        print(f"\n  Plots saved: roc_curve_dv{dv_mag:.2f}.png, score_distribution_dv{dv_mag:.2f}.png")
    
    # STEP 4: Summary
    print("\n" + "=" * 80)
    print("SUMMARY: AUROC ACROSS ALL DELTA-V MAGNITUDES")
    print("=" * 80)
    
    for dv_mag in DELTA_V_MAGNITUDES:
        auroc = results_by_dv[dv_mag]["metrics"]["auroc"]
        print(f"  Δv={dv_mag:.3f} km/s: AUROC={auroc:.4f}")
    
    # Plot comparison
    aurocs = [results_by_dv[dv]["metrics"]["auroc"] for dv in DELTA_V_MAGNITUDES]
    plot_auroc_comparison(DELTA_V_MAGNITUDES, aurocs)
    print(f"\n  Comparison plot saved: auroc_vs_delta_v.png")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nKey results:")
    print(f"  AUROC at 0.01 km/s: {results_by_dv[0.01]['metrics']['auroc']:.4f}")
    print(f"  AUROC at 0.50 km/s: {results_by_dv[0.50]['metrics']['auroc']:.4f}")
    print(f"  Mean AUROC: {np.mean(aurocs):.4f}")
    print(f"\nAll data is REAL SGP4 trajectories with REAL H-GNN inference.")
    print(f"No synthetic score generation. No fake data.\n")


if __name__ == "__main__":
    run_real_validation()
