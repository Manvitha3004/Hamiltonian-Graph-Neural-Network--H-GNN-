"""Shared pytest fixtures for anomaly-related tests."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


class DummyHamiltonianModel(nn.Module):
    """Minimal model exposing the API needed by anomaly scoring tests."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.hamiltonian = SimpleNamespace(Phi=None)

    def compute_hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # Deterministic scalar energy proxy for analytical expected-value checks.
        return q.sum() + 2.0 * p.sum() + 0.0 * self.anchor


@pytest.fixture
def dummy_hamiltonian_model() -> DummyHamiltonianModel:
    """Provide a deterministic model stub for anomaly score tests."""
    return DummyHamiltonianModel()


@pytest.fixture
def zero_state_nt() -> tuple[torch.Tensor, torch.Tensor]:
    """Provide zero trajectories with shape (N, T, 3)."""
    q = torch.zeros(2, 4, 3)
    p = torch.zeros(2, 4, 3)
    return q, p
