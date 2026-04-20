"""Utilities for synthetic maneuver anomaly generation.

This module provides:
1) Impulsive maneuver injection into trajectory tensors.
2) Anomaly dataset generation with labels and maneuver metadata.
3) Optional trajectory window extraction and Dataset wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ManeuverMetadata:
    """Metadata for an injected maneuver event."""

    satellite_index: int
    t_maneuver: int
    delta_v_vector: List[float]


def _to_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert numpy input to torch tensor while preserving torch inputs."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asarray(x))


def inject_maneuver(
    positions: torch.Tensor | np.ndarray,
    velocities: torch.Tensor | np.ndarray,
    t_maneuver: int,
    delta_v: torch.Tensor | np.ndarray,
    satellite_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inject an impulsive maneuver into a trajectory.

    Args:
        positions: Position tensor with shape (N, T, 3).
        velocities: Velocity tensor with shape (N, T, 3).
        t_maneuver: Timestep index where maneuver starts.
        delta_v: Delta-v vector in km/s, shape (3,).
        satellite_index: Satellite index receiving the impulse.

    Returns:
        Tuple of cloned (positions, velocities) with injected maneuver.

    Notes:
        The impulse is applied to all timesteps t >= t_maneuver for the selected
        satellite to represent a velocity discontinuity that persists afterward.
    """
    q = _to_tensor(positions).clone()
    p = _to_tensor(velocities).clone()

    if q.ndim != 3 or p.ndim != 3:
        raise ValueError("Expected positions/velocities with shape (N, T, 3).")
    if q.shape != p.shape:
        raise ValueError("positions and velocities must have identical shape.")
    if q.shape[-1] != 3:
        raise ValueError("Last dimension must be 3 for 3D state vectors.")

    n_sat, n_steps, _ = q.shape
    if satellite_index < 0 or satellite_index >= n_sat:
        raise IndexError("satellite_index out of range.")
    if t_maneuver < 0 or t_maneuver >= n_steps:
        raise IndexError("t_maneuver out of range.")

    dv = _to_tensor(delta_v).to(dtype=p.dtype, device=p.device).view(1, 1, 3)
    p[satellite_index, t_maneuver:, :] = p[satellite_index, t_maneuver:, :] + dv

    return q, p


def build_trajectory_windows(
    positions: torch.Tensor | np.ndarray,
    velocities: torch.Tensor | np.ndarray,
    window_size: int,
    stride: int,
) -> Dict[str, torch.Tensor]:
    """Create fixed-length trajectory windows from a single multi-satellite sequence.

    Args:
        positions: Tensor with shape (N, T, 3).
        velocities: Tensor with shape (N, T, 3).
        window_size: Number of timesteps per window.
        stride: Step between window starts.

    Returns:
        Dict with keys:
            positions: (M, N, window_size, 3)
            velocities: (M, N, window_size, 3)
        where M is number of windows.
    """
    q = _to_tensor(positions)
    p = _to_tensor(velocities)

    if q.shape != p.shape:
        raise ValueError("positions and velocities must have identical shape.")
    if q.ndim != 3 or q.shape[-1] != 3:
        raise ValueError("Expected shape (N, T, 3).")
    if window_size < 2:
        raise ValueError("window_size must be >= 2.")

    n_sat, total_steps, _ = q.shape
    _ = n_sat

    if total_steps < window_size:
        raise ValueError("window_size cannot exceed available timesteps.")

    q_windows: List[torch.Tensor] = []
    p_windows: List[torch.Tensor] = []

    for start in range(0, total_steps - window_size + 1, max(1, stride)):
        end = start + window_size
        q_windows.append(q[:, start:end, :].clone())
        p_windows.append(p[:, start:end, :].clone())

    return {
        "positions": torch.stack(q_windows, dim=0),
        "velocities": torch.stack(p_windows, dim=0),
    }


def generate_anomaly_dataset(
    normal_trajectories: Dict[str, torch.Tensor | np.ndarray],
    anomaly_rate: float = 0.3,
    rng_seed: Optional[int] = 42,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[Dict[str, Any]]]:
    """Generate labeled anomaly trajectories by synthetic maneuver injection.

    Args:
        normal_trajectories: Dict with keys:
            positions: (M, N, T, 3)
            velocities: (M, N, T, 3)
        anomaly_rate: Fraction of trajectories labeled anomalous.
        rng_seed: Optional seed for reproducibility.

    Returns:
        trajectories: Dict with keys `positions`, `velocities`, each (M, N, T, 3)
        labels: Tensor (M,) with 0=nominal, 1=anomalous
        maneuver_metadata: List of length M with maneuver details per sample.
            For nominal samples: {satellite_index: -1, t_maneuver: -1, delta_v_vector: [0,0,0]}
    """
    if not (0.0 <= anomaly_rate <= 1.0):
        raise ValueError("anomaly_rate must be in [0, 1].")

    q_all = _to_tensor(normal_trajectories["positions"]).clone()
    p_all = _to_tensor(normal_trajectories["velocities"]).clone()

    if q_all.shape != p_all.shape:
        raise ValueError("positions and velocities must have identical shape.")
    if q_all.ndim != 4 or q_all.shape[-1] != 3:
        raise ValueError("Expected trajectories with shape (M, N, T, 3).")

    n_samples, n_sat, n_steps, _ = q_all.shape

    rng = np.random.default_rng(rng_seed)
    n_anomalies = int(round(anomaly_rate * n_samples))
    anomaly_indices = set(rng.choice(n_samples, size=n_anomalies, replace=False).tolist())

    labels = torch.zeros(n_samples, dtype=torch.float32)
    metadata: List[Dict[str, Any]] = []

    t_min = max(1, int(0.25 * n_steps))
    t_max = max(t_min + 1, int(0.75 * n_steps))

    for idx in range(n_samples):
        if idx not in anomaly_indices:
            metadata.append(
                {
                    "satellite_index": -1,
                    "t_maneuver": -1,
                    "delta_v_vector": [0.0, 0.0, 0.0],
                }
            )
            continue

        sat_idx = int(rng.integers(0, n_sat))
        t_maneuver = int(rng.integers(t_min, t_max))

        dv_mag = float(rng.uniform(0.01, 0.5))
        dv_dir = rng.normal(size=3)
        dv_dir /= (np.linalg.norm(dv_dir) + 1e-12)
        dv_vec = dv_mag * dv_dir

        q_i, p_i = inject_maneuver(
            q_all[idx],
            p_all[idx],
            t_maneuver=t_maneuver,
            delta_v=dv_vec,
            satellite_index=sat_idx,
        )

        q_all[idx] = q_i
        p_all[idx] = p_i
        labels[idx] = 1.0

        metadata.append(
            {
                "satellite_index": sat_idx,
                "t_maneuver": t_maneuver,
                "delta_v_vector": [float(v) for v in dv_vec.tolist()],
            }
        )

    return {"positions": q_all, "velocities": p_all}, labels, metadata


class AnomalyTrajectoryDataset(Dataset):
    """Torch dataset for anomaly detection trajectories.

    Each item returns:
        positions: (N, T, 3)
        velocities: (N, T, 3)
        label: scalar float tensor
    """

    def __init__(
        self,
        trajectories: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ):
        self.positions = trajectories["positions"].float()
        self.velocities = trajectories["velocities"].float()
        self.labels = labels.float()

        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have identical shape.")
        if self.positions.shape[0] != self.labels.shape[0]:
            raise ValueError("Label count must match number of trajectories.")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "positions": self.positions[index],
            "velocities": self.velocities[index],
            "label": self.labels[index],
        }
