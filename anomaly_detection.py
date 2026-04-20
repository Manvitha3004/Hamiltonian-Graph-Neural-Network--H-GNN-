"""Anomaly detection utilities using Hamiltonian residual signals.

This module adds a second-stage detector on top of a frozen H-GNN by using:
1) trajectory residuals in phase space,
2) Hamiltonian energy residuals,
3) summary statistics of learned pairwise interaction magnitudes.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def _get_model_device_dtype(model: nn.Module) -> Tuple[torch.device, torch.dtype]:
    """Infer device and dtype from the first model parameter."""
    param = next(model.parameters())
    return param.device, param.dtype


def _as_time_major(state: torch.Tensor) -> torch.Tensor:
    """Convert state tensor to shape (T, N, 3).

    Accepts either (N, T, 3) or (T, N, 3).
    """
    if state.ndim != 3 or state.shape[-1] != 3:
        raise ValueError("State tensor must have shape (N, T, 3) or (T, N, 3).")

    # Heuristic: if first dim is usually satellite count and second dim is time.
    # For this codebase, N << T for most datasets, so transpose when dim0 < dim1.
    if state.shape[0] < state.shape[1]:
        return state.permute(1, 0, 2).contiguous()
    return state


def _unpack_trajectory(traj: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack trajectory input into (q, p) tensors."""
    if isinstance(traj, dict):
        q = traj["positions"]
        p = traj["velocities"]
    elif isinstance(traj, (tuple, list)) and len(traj) == 2:
        q, p = traj
    else:
        raise ValueError("Trajectory must be dict with positions/velocities or a (q, p) tuple.")

    if not isinstance(q, torch.Tensor):
        q = torch.as_tensor(q)
    if not isinstance(p, torch.Tensor):
        p = torch.as_tensor(p)

    return _as_time_major(q), _as_time_major(p)


def predict_trajectory(
    model: nn.Module,
    actual_traj: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict a full trajectory from the first state using the H-GNN model.

    Args:
        model: Trained H-GNN model.
        actual_traj: Actual trajectory in dict or tuple format.

    Returns:
        (q_pred, p_pred) in shape (T, N, 3).
    """
    q_actual, p_actual = _unpack_trajectory(actual_traj)
    device, dtype = _get_model_device_dtype(model)

    q_actual = q_actual.to(device=device, dtype=dtype)
    p_actual = p_actual.to(device=device, dtype=dtype)

    q0 = q_actual[0]
    p0 = p_actual[0]
    n_steps = int(q_actual.shape[0] - 1)

    # Assumption: HGNN.forward(q, p, n_steps, return_trajectory=True)
    # returns (q_traj, p_traj) with shape (n_steps + 1, N, 3).
    with torch.no_grad():
        q_pred, p_pred = model(q0, p0, n_steps=n_steps, return_trajectory=True)

    return q_pred, p_pred


def _compute_phi_norm_summary(
    model: nn.Module,
    q_traj: torch.Tensor,
    p_traj: torch.Tensor,
) -> Dict[str, float]:
    """Compute mean/max/std of |Phi_ij| over trajectory edges and timesteps."""
    phi_module = getattr(model.hamiltonian, "Phi", None)
    if phi_module is None:
        return {"mean": 0.0, "max": 0.0, "std": 0.0}

    phi_vals: List[torch.Tensor] = []
    with torch.no_grad():
        for t in range(q_traj.shape[0]):
            q_t = q_traj[t]
            p_t = p_traj[t]
            graph = model.graph_builder(q_t)
            edge_index = graph.edge_index

            if edge_index.numel() == 0:
                continue

            src, dst = edge_index
            phi_ij = phi_module(q_t[src], p_t[src], q_t[dst], p_t[dst])
            phi_vals.append(torch.abs(phi_ij).reshape(-1))

    if not phi_vals:
        return {"mean": 0.0, "max": 0.0, "std": 0.0}

    all_phi = torch.cat(phi_vals)
    return {
        "mean": float(all_phi.mean().item()),
        "max": float(all_phi.max().item()),
        "std": float(all_phi.std(unbiased=False).item()),
    }


def compute_residual_components(
    model: nn.Module,
    actual_traj: Any,
    predicted_traj: Any,
    q_scale: float = 10000.0,
    p_scale: float = 10.0,
) -> Dict[str, Any]:
    """Compute residual signals and aggregate statistics for anomaly detection.

    NORMALIZED residuals (scaled to [0, ~10] range):
    residual(t) = ||q_actual(t) - q_pred(t)||^2 / q_scale^2
                + ||p_actual(t) - p_pred(t)||^2 / p_scale^2

    RELATIVE anomaly score:
    baseline = mean residual over first 10 timesteps
    relative_residual(t) = residual(t) / (baseline + 1e-8)
    anomaly_score = max_t relative_residual(t)

    This isolates the maneuver spike from baseline orbit error.
    """
    q_actual, p_actual = _unpack_trajectory(actual_traj)
    q_pred, p_pred = _unpack_trajectory(predicted_traj)

    if q_actual.shape != q_pred.shape or p_actual.shape != p_pred.shape:
        raise ValueError("Actual and predicted trajectories must share the same shape.")

    device, dtype = _get_model_device_dtype(model)
    q_actual = q_actual.to(device=device, dtype=dtype)
    p_actual = p_actual.to(device=device, dtype=dtype)
    q_pred = q_pred.to(device=device, dtype=dtype)
    p_pred = p_pred.to(device=device, dtype=dtype)

    dq = q_actual - q_pred
    dp = p_actual - p_pred

    # FIX 1: Normalize residuals to physical scale
    # Position component: ||dq||^2 / (10000 km)^2
    # Velocity component: ||dp||^2 / (10 km/s)^2
    q_residual_norm = torch.sum(dq * dq, dim=(1, 2)) / (q_scale ** 2)
    p_residual_norm = torch.sum(dp * dp, dim=(1, 2)) / (p_scale ** 2)
    residual_series = q_residual_norm + p_residual_norm

    # FIX 2: Compute baseline from first 10 timesteps
    n_baseline = min(10, max(1, residual_series.shape[0] // 2))
    baseline_residual = float(residual_series[:n_baseline].mean().item())
    baseline_residual = max(baseline_residual, 1e-8)  # Avoid division by zero

    # FIX 3: Compute relative residuals and take max
    relative_residual_series = residual_series / baseline_residual
    max_relative_residual = float(relative_residual_series.max().item())

    # Energy residuals (keep for feature extraction, but not in anomaly_score)
    energy_residual_list: List[torch.Tensor] = []
    with torch.no_grad():
        for t in range(q_actual.shape[0]):
            H_actual = model.compute_hamiltonian(q_actual[t], p_actual[t])
            H_pred = model.compute_hamiltonian(q_pred[t], p_pred[t])
            energy_residual_list.append(torch.abs(H_actual - H_pred).reshape(1))

    energy_residual_series = torch.cat(energy_residual_list, dim=0)
    mean_energy_residual = float(energy_residual_series.mean().item())

    # Anomaly score = max relative residual (anomaly spike detection)
    anomaly_score = max_relative_residual

    phi_summary = _compute_phi_norm_summary(model, q_actual, p_actual)

    return {
        "residual_series": residual_series,
        "relative_residual_series": relative_residual_series,
        "energy_residual_series": energy_residual_series,
        "max_residual": float(residual_series.max().item()),
        "max_relative_residual": max_relative_residual,
        "baseline_residual": baseline_residual,
        "mean_energy_residual": mean_energy_residual,
        "anomaly_score": float(anomaly_score),
        "phi_summary": phi_summary,
    }


def compute_anomaly_score(
    model: nn.Module,
    actual_traj: Any,
    predicted_traj: Any,
) -> float:
    """Compute scalar anomaly score from trajectory and energy residuals."""
    components = compute_residual_components(model, actual_traj, predicted_traj)
    return float(components["anomaly_score"])


def extract_detection_features(
    model: nn.Module,
    actual_traj: Any,
    predicted_traj: Any,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Create the 6D feature vector for anomaly classifier.

    Features (normalized and relative):
        [anomaly_score, max_relative_residual, mean_energy_residual,
         mean_Phi_norm, max_Phi_norm, std_Phi_norm]
    """
    comp = compute_residual_components(model, actual_traj, predicted_traj)
    phi_summary = comp["phi_summary"]

    feats = torch.tensor(
        [
            comp["anomaly_score"],
            comp["max_relative_residual"],
            comp["mean_energy_residual"],
            phi_summary["mean"],
            phi_summary["max"],
            phi_summary["std"],
        ],
        dtype=torch.float32,
    )
    return feats, comp


class AnomalyDetectionHead(nn.Module):
    """Lightweight MLP detection head for trajectory-level anomaly probability."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feature tensor of shape (B, 6) or (6,)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)


def _iter_batch_samples(batch: Dict[str, torch.Tensor]) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield per-sample tensors from a collated dataloader batch dict."""
    q_batch = batch["positions"]
    p_batch = batch["velocities"]
    y_batch = batch["label"]

    if q_batch.ndim == 3:
        q_batch = q_batch.unsqueeze(0)
        p_batch = p_batch.unsqueeze(0)
        y_batch = y_batch.unsqueeze(0)

    for i in range(q_batch.shape[0]):
        yield q_batch[i], p_batch[i], y_batch[i]


def train_detection_head(
    hgnn_model: nn.Module,
    detection_head: nn.Module,
    anomaly_dataloader: DataLoader,
    epochs: int = 20,
) -> Dict[str, List[float]]:
    """Train detection head while keeping base H-GNN frozen.

    Args:
        hgnn_model: Trained HGNN model (frozen in this phase).
        detection_head: Detection MLP head.
        anomaly_dataloader: DataLoader yielding positions/velocities/label.
        epochs: Number of training epochs.

    Returns:
        Dict containing per-epoch loss history.
    """
    for p in hgnn_model.parameters():
        p.requires_grad = False
    hgnn_model.eval()

    device, _ = _get_model_device_dtype(hgnn_model)
    detection_head.to(device)
    detection_head.train()

    optimizer = torch.optim.AdamW(detection_head.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    history: Dict[str, List[float]] = {"loss": []}

    for epoch in range(epochs):
        epoch_losses: List[float] = []

        for batch in anomaly_dataloader:
            feature_list: List[torch.Tensor] = []
            label_list: List[torch.Tensor] = []

            for q_i, p_i, y_i in _iter_batch_samples(batch):
                actual = {"positions": q_i, "velocities": p_i}
                q_pred, p_pred = predict_trajectory(hgnn_model, actual)
                feats, _ = extract_detection_features(
                    hgnn_model,
                    actual,
                    {"positions": q_pred, "velocities": p_pred},
                )
                feature_list.append(feats)
                label_list.append(y_i.float().reshape(1))

            x = torch.stack(feature_list, dim=0).to(device)
            y = torch.stack(label_list, dim=0).to(device)

            optimizer.zero_grad()
            y_hat = detection_head(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        history["loss"].append(float(np.mean(epoch_losses) if epoch_losses else 0.0))

    return history


def _compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC with trapezoidal integration, no sklearn dependency."""
    y_true = y_true.astype(np.int64)
    if len(np.unique(y_true)) < 2:
        return float("nan")

    order = np.argsort(-y_score)
    y_true = y_true[order]

    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)

    tp = 0.0
    fp = 0.0
    tpr = [0.0]
    fpr = [0.0]

    for label in y_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / max(pos, 1))
        fpr.append(fp / max(neg, 1))

    return float(np.trapz(np.array(tpr), np.array(fpr)))


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute precision/recall/F1 for binary predictions."""
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_anomaly_detection(
    model: nn.Module,
    detection_head: nn.Module,
    test_loader: DataLoader,
) -> Dict[str, float]:
    """Evaluate anomaly detection metrics and save ROC/distribution plots.

    Returns:
        Dict containing AUROC, precision, recall, F1, and classwise average scores.
    """
    model.eval()
    detection_head.eval()

    device, _ = _get_model_device_dtype(model)

    labels: List[float] = []
    probs: List[float] = []
    anomaly_scores: List[float] = []

    with torch.no_grad():
        for batch in test_loader:
            for q_i, p_i, y_i in _iter_batch_samples(batch):
                actual = {"positions": q_i, "velocities": p_i}
                q_pred, p_pred = predict_trajectory(model, actual)
                feats, comp = extract_detection_features(
                    model,
                    actual,
                    {"positions": q_pred, "velocities": p_pred},
                )

                prob = detection_head(feats.to(device)).squeeze().item()

                labels.append(float(y_i.item()))
                probs.append(float(prob))
                anomaly_scores.append(float(comp["anomaly_score"]))

    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    scores = np.asarray(anomaly_scores, dtype=np.float64)

    auroc = _compute_auroc(y_true, y_prob)
    cls_metrics = _binary_metrics(y_true, y_prob, threshold=0.5)

    nominal_scores = scores[y_true == 0]
    anomalous_scores = scores[y_true == 1]

    nominal_mean = float(np.mean(nominal_scores)) if nominal_scores.size else float("nan")
    anomalous_mean = float(np.mean(anomalous_scores)) if anomalous_scores.size else float("nan")

    # ROC curve
    plt.figure(figsize=(6, 5))
    if len(np.unique(y_true)) >= 2:
        order = np.argsort(-y_prob)
        y_sorted = y_true[order]
        pos = np.sum(y_sorted == 1)
        neg = np.sum(y_sorted == 0)
        tp = 0.0
        fp = 0.0
        tpr = [0.0]
        fpr = [0.0]
        for label in y_sorted:
            if label == 1:
                tp += 1.0
            else:
                fp += 1.0
            tpr.append(tp / max(pos, 1))
            fpr.append(fp / max(neg, 1))
        plt.plot(fpr, tpr, label=f"ROC (AUROC={auroc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Anomaly Detection ROC")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("anomaly_roc_curve.png", dpi=150)
    plt.close()

    # Score distribution
    plt.figure(figsize=(7, 5))
    if nominal_scores.size:
        plt.hist(nominal_scores, bins=30, alpha=0.6, label="Nominal", density=True)
    if anomalous_scores.size:
        plt.hist(anomalous_scores, bins=30, alpha=0.6, label="Anomalous", density=True)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.title("Anomaly Score Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("anomaly_score_distribution.png", dpi=150)
    plt.close()

    return {
        "auroc": float(auroc),
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
        "avg_anomaly_score_nominal": nominal_mean,
        "avg_anomaly_score_anomalous": anomalous_mean,
    }
