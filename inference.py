"""Inference API for maneuver anomaly detection."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from anomaly_detection import (
    extract_detection_features,
    predict_trajectory,
)


def detect_maneuver(
    model: nn.Module,
    detection_head: nn.Module,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Detect anomalous maneuvers from a live trajectory segment.

    Args:
        model: Trained H-GNN model.
        detection_head: Trained anomaly detection head.
        positions: Position trajectory, shape (N, T, 3) or (T, N, 3), in km.
        velocities: Velocity trajectory, shape (N, T, 3) or (T, N, 3), in km/s.
        threshold: Decision threshold on predicted anomaly probability.

    Returns:
        Dictionary:
            is_anomalous: bool
            anomaly_probability: float
            anomaly_score: float
            flagged_timesteps: list[int]
            phi_summary: {mean, max, std}
    """
    model.eval()
    detection_head.eval()

    actual = {"positions": positions, "velocities": velocities}

    with torch.no_grad():
        q_pred, p_pred = predict_trajectory(model, actual)
        features, details = extract_detection_features(
            model,
            actual,
            {"positions": q_pred, "velocities": p_pred},
        )

        param = next(model.parameters())
        probability = float(
            detection_head(features.to(device=param.device)).squeeze().item()
        )

    residual_series = details["residual_series"].detach().cpu().numpy()
    mean_residual = float(residual_series.mean())
    flagged: List[int] = [
        int(t)
        for t, r_t in enumerate(residual_series)
        if float(r_t) > 2.0 * mean_residual
    ]

    return {
        "is_anomalous": bool(probability >= threshold),
        "anomaly_probability": probability,
        "anomaly_score": float(details["anomaly_score"]),
        "flagged_timesteps": flagged,
        "phi_summary": {
            "mean": float(details["phi_summary"]["mean"]),
            "max": float(details["phi_summary"]["max"]),
            "std": float(details["phi_summary"]["std"]),
        },
    }
