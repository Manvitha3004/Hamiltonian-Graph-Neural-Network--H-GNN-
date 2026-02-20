"""
Tests for Hamiltonian module.

Validates:
1. Energy conservation for two-body problem
2. Gradient computation via autograd
3. SO(3) invariance of Hamiltonian
"""

import pytest
import torch
import numpy as np

from h_gnn.models.hamiltonian import (
    TwoBodyHamiltonian,
    KeplerianHamiltonian,
    GraphHamiltonian,
    compute_angular_momentum,
    compute_eccentricity_vector,
)


class TestTwoBodyHamiltonian:
    """Tests for TwoBodyHamiltonian."""
    
    @pytest.fixture
    def hamiltonian(self):
        return TwoBodyHamiltonian(mu=398600.4418)
    
    @pytest.fixture
    def circular_orbit(self):
        """Generate circular orbit initial conditions."""
        mu = 398600.4418
        R = 6378.137 + 400  # 400 km altitude
        v = np.sqrt(mu / R)
        
        q = torch.tensor([[R, 0.0, 0.0]], dtype=torch.float64)
        p = torch.tensor([[0.0, v, 0.0]], dtype=torch.float64)
        
        return q, p
    
    def test_energy_value(self, hamiltonian, circular_orbit):
        """Test energy computation gives correct value."""
        q, p = circular_orbit
        
        H = hamiltonian(q, p, sum_over_particles=True)
        
        # For circular orbit: E = -μ/(2a) = -μ/(2R)
        R = q.norm().item()
        mu = 398600.4418
        expected_E = -mu / (2 * R)
        
        assert torch.isclose(H, torch.tensor(expected_E, dtype=torch.float64), rtol=1e-6)
    
    def test_kinetic_energy(self, hamiltonian):
        """Test kinetic energy computation."""
        p = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float64)
        
        T = hamiltonian.kinetic_energy(p)
        
        expected = 0.5 * (9 + 16)  # 0.5 * |p|²
        assert torch.isclose(T, torch.tensor([expected], dtype=torch.float64))
    
    def test_potential_energy(self, hamiltonian):
        """Test potential energy computation."""
        q = torch.tensor([[7000.0, 0.0, 0.0]], dtype=torch.float64)
        
        V = hamiltonian.potential_energy(q)
        
        expected = -398600.4418 / 7000.0
        assert torch.isclose(V, torch.tensor([expected], dtype=torch.float64), rtol=1e-6)
    
    def test_gradient_computation(self, hamiltonian, circular_orbit):
        """Test gradient computation via autograd."""
        q, p = circular_orbit
        
        grad_q, grad_p = hamiltonian.compute_gradients(q, p)
        
        # ∂H/∂p = p (for kinetic energy 0.5|p|²)
        assert torch.allclose(grad_p, p, rtol=1e-6)
        
        # ∂H/∂q = μq/|q|³ (gravitational force direction)
        r = q.norm()
        expected_grad_q = 398600.4418 * q / (r ** 3)
        assert torch.allclose(grad_q, expected_grad_q, rtol=1e-5)
    
    def test_so3_invariance(self, hamiltonian, circular_orbit):
        """Test SO(3) invariance: H(Rq, Rp) = H(q, p)."""
        q, p = circular_orbit
        
        H_original = hamiltonian(q, p, sum_over_particles=True)
        
        # Random rotation
        from scipy.spatial.transform import Rotation
        R = torch.tensor(Rotation.random().as_matrix(), dtype=torch.float64)
        
        q_rot = torch.matmul(q, R.T)
        p_rot = torch.matmul(p, R.T)
        
        H_rotated = hamiltonian(q_rot, p_rot, sum_over_particles=True)
        
        assert torch.isclose(H_original, H_rotated, rtol=1e-10)
    
    def test_vector_field(self, hamiltonian, circular_orbit):
        """Test Hamiltonian vector field computation."""
        q, p = circular_orbit
        
        dq_dt, dp_dt = hamiltonian.hamiltonian_vector_field(q, p)
        
        # dq/dt = p
        assert torch.allclose(dq_dt, p, rtol=1e-6)
        
        # dp/dt = -∂H/∂q
        r = q.norm()
        expected_dp = 398600.4418 * q / (r ** 3)  # Gravitational acceleration
        assert torch.allclose(-dp_dt, expected_dp, rtol=1e-5)


class TestAngularMomentum:
    """Tests for angular momentum computation."""
    
    def test_angular_momentum_value(self):
        """Test L = q × p computation."""
        q = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        p = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        
        L = compute_angular_momentum(q, p)
        
        # 1×0 - 0×1 = 0, 0×0 - 1×0 = 0, 1×1 - 0×0 = 1
        expected = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        assert torch.allclose(L, expected)
    
    def test_angular_momentum_perpendicular(self):
        """Test L is perpendicular to both q and p."""
        q = torch.tensor([[7000.0, 0.0, 0.0]], dtype=torch.float64)
        p = torch.tensor([[0.0, 7.5, 0.0]], dtype=torch.float64)
        
        L = compute_angular_momentum(q, p)
        
        # L · q = 0
        assert torch.isclose(torch.sum(L * q), torch.tensor(0.0, dtype=torch.float64), atol=1e-10)
        
        # L · p = 0
        assert torch.isclose(torch.sum(L * p), torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


class TestEccentricityVector:
    """Tests for eccentricity vector computation."""
    
    def test_circular_orbit_eccentricity(self):
        """Test that circular orbit has e ≈ 0."""
        mu = 398600.4418
        R = 7000.0
        v = np.sqrt(mu / R)
        
        q = torch.tensor([[R, 0.0, 0.0]], dtype=torch.float64)
        p = torch.tensor([[0.0, v, 0.0]], dtype=torch.float64)
        
        e = compute_eccentricity_vector(q, p, mu)
        e_mag = torch.norm(e)
        
        assert e_mag.item() < 0.01  # Should be ~0 for circular orbit


class TestGraphHamiltonian:
    """Tests for GraphHamiltonian."""
    
    @pytest.fixture
    def orbit_hamiltonian(self):
        return TwoBodyHamiltonian(mu=398600.4418)
    
    def test_lambda_zero_recovers_twobody(self, orbit_hamiltonian):
        """Test that λ=0 recovers pure two-body dynamics."""
        from h_gnn.models.interaction import InteractionPotential
        
        interaction = InteractionPotential(hidden_dims=[32, 32])
        graph_H = GraphHamiltonian(
            orbit_hamiltonian,
            interaction_potential=interaction,
            interaction_strength=0.0,  # λ = 0
        )
        
        q = torch.randn(10, 3, dtype=torch.float64) * 7000
        p = torch.randn(10, 3, dtype=torch.float64) * 10
        
        # With λ=0, should equal orbital Hamiltonian
        H_graph = graph_H(q, p, edge_index=None)
        H_orbit = orbit_hamiltonian(q, p, sum_over_particles=True)
        
        assert torch.isclose(H_graph, H_orbit, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
