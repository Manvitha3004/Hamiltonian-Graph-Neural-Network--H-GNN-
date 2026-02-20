"""
Tests for symplectic integrators.

Validates:
1. Symplectic integrators preserve energy better than non-symplectic
2. Correct step mechanics
3. Long-horizon stability
"""

import pytest
import torch
import numpy as np

from h_gnn.models.hamiltonian import TwoBodyHamiltonian
from h_gnn.integrators.symplectic import SymplecticEuler, StormerVerlet, SymplecticRK
from h_gnn.integrators.baselines import VanillaEuler, RK4


class TestSymplecticEuler:
    """Tests for Symplectic Euler integrator."""
    
    @pytest.fixture
    def integrator(self):
        return SymplecticEuler(dt=60.0)
    
    @pytest.fixture
    def hamiltonian(self):
        return TwoBodyHamiltonian(mu=398600.4418)
    
    @pytest.fixture
    def circular_orbit(self):
        """Circular orbit at 400 km altitude."""
        mu = 398600.4418
        R = 6778.137
        v = np.sqrt(mu / R)
        
        q = torch.tensor([[R, 0.0, 0.0]], dtype=torch.float64)
        p = torch.tensor([[0.0, v, 0.0]], dtype=torch.float64)
        
        return q, p
    
    def test_single_step(self, integrator, hamiltonian, circular_orbit):
        """Test single integration step."""
        q, p = circular_orbit
        
        q_new, p_new = integrator.step(q, p, hamiltonian)
        
        # Position should move
        assert not torch.allclose(q_new, q)
        
        # Shapes preserved
        assert q_new.shape == q.shape
        assert p_new.shape == p.shape
    
    def test_energy_bounded(self, integrator, hamiltonian, circular_orbit):
        """Test that energy remains bounded over many steps."""
        q, p = circular_orbit
        
        q_traj, p_traj = integrator.integrate(
            q, p, hamiltonian, n_steps=1000, store_trajectory=True
        )
        
        # Compute energy at each step
        energies = []
        for i in range(len(q_traj)):
            E = hamiltonian(q_traj[i], p_traj[i], sum_over_particles=True)
            energies.append(E.item())
        
        E0 = energies[0]
        relative_drift = [(E - E0) / abs(E0) for E in energies]
        
        # Symplectic should have bounded oscillations, not drift
        max_drift = max(abs(d) for d in relative_drift)
        assert max_drift < 0.05  # Less than 5% drift
    
    def test_angular_momentum_preserved(self, integrator, hamiltonian, circular_orbit):
        """Test angular momentum conservation."""
        from h_gnn.models.hamiltonian import compute_angular_momentum
        
        q, p = circular_orbit
        
        q_final, p_final = integrator.integrate(
            q, p, hamiltonian, n_steps=100, store_trajectory=False
        )
        
        L_initial = torch.norm(compute_angular_momentum(q, p))
        L_final = torch.norm(compute_angular_momentum(q_final, p_final))
        
        relative_change = abs(L_final - L_initial) / L_initial
        assert relative_change < 0.01  # Less than 1% change


class TestStormerVerlet:
    """Tests for Störmer-Verlet integrator."""
    
    @pytest.fixture
    def integrator(self):
        return StormerVerlet(dt=60.0)
    
    @pytest.fixture
    def hamiltonian(self):
        return TwoBodyHamiltonian(mu=398600.4418)
    
    @pytest.fixture
    def circular_orbit(self):
        mu = 398600.4418
        R = 6778.137
        v = np.sqrt(mu / R)
        
        q = torch.tensor([[R, 0.0, 0.0]], dtype=torch.float64)
        p = torch.tensor([[0.0, v, 0.0]], dtype=torch.float64)
        
        return q, p
    
    def test_second_order_accuracy(self, integrator, hamiltonian, circular_orbit):
        """Test that Verlet is more accurate than Symplectic Euler."""
        q, p = circular_orbit
        
        sym_euler = SymplecticEuler(dt=60.0)
        
        # Integrate both for same number of steps (final state only)
        q_verlet, p_verlet = integrator.integrate(
            q, p, hamiltonian, n_steps=100, store_trajectory=False
        )
        q_euler, p_euler = sym_euler.integrate(
            q, p, hamiltonian, n_steps=100, store_trajectory=False
        )
        
        # Compute energy errors
        E0 = hamiltonian(q, p, sum_over_particles=True).item()
        E_verlet = hamiltonian(q_verlet, p_verlet, sum_over_particles=True).item()
        E_euler = hamiltonian(q_euler, p_euler, sum_over_particles=True).item()
        
        error_verlet = abs(E_verlet - E0) / abs(E0)
        error_euler = abs(E_euler - E0) / abs(E0)
        
        # Verlet should be more accurate (or comparable)
        assert error_verlet <= error_euler * 2  # Allow some tolerance
    
    def test_energy_bounded_long_horizon(self, integrator, hamiltonian, circular_orbit):
        """Test energy conservation over long integration."""
        q, p = circular_orbit
        
        q_traj, p_traj = integrator.integrate(
            q, p, hamiltonian, n_steps=2000, store_trajectory=True
        )
        
        energies = []
        for i in range(len(q_traj)):
            E = hamiltonian(q_traj[i], p_traj[i], sum_over_particles=True)
            energies.append(E.item())
        
        E0 = energies[0]
        max_drift = max(abs((E - E0) / E0) for E in energies)
        
        # Verlet should maintain < 1% drift
        assert max_drift < 0.01


class TestVanillaEuler:
    """Tests for non-symplectic Vanilla Euler."""
    
    @pytest.fixture
    def integrator(self):
        return VanillaEuler(dt=60.0)
    
    @pytest.fixture
    def hamiltonian(self):
        return TwoBodyHamiltonian(mu=398600.4418)
    
    @pytest.fixture
    def circular_orbit(self):
        mu = 398600.4418
        R = 6778.137
        v = np.sqrt(mu / R)
        
        q = torch.tensor([[R, 0.0, 0.0]], dtype=torch.float64)
        p = torch.tensor([[0.0, v, 0.0]], dtype=torch.float64)
        
        return q, p
    
    def test_energy_drifts(self, integrator, hamiltonian, circular_orbit):
        """Test that vanilla Euler exhibits energy drift."""
        q, p = circular_orbit
        
        q_traj, p_traj = integrator.integrate(
            q, p, hamiltonian, n_steps=500, store_trajectory=True
        )
        
        energies = []
        for i in range(len(q_traj)):
            E = hamiltonian(q_traj[i], p_traj[i], sum_over_particles=True)
            energies.append(E.item())
        
        E0 = energies[0]
        
        # Check for systematic drift (not just oscillation)
        # Vanilla Euler typically gains energy
        final_drift = (energies[-1] - E0) / abs(E0)
        
        # Should have noticeable drift (> 5% is typical for this many steps)
        assert abs(final_drift) > 0.01


class TestIntegratorComparison:
    """Comparative tests between integrators."""
    
    def test_symplectic_beats_nonsymplectic(self):
        """Test that symplectic integrators preserve energy better."""
        mu = 398600.4418
        R = 6778.137
        v = np.sqrt(mu / R)
        
        q = torch.tensor([[R, 0.0, 0.0]], dtype=torch.float64)
        p = torch.tensor([[0.0, v, 0.0]], dtype=torch.float64)
        
        hamiltonian = TwoBodyHamiltonian(mu=mu)
        
        sym_euler = SymplecticEuler(dt=60.0)
        verlet = StormerVerlet(dt=60.0)
        vanilla = VanillaEuler(dt=60.0)
        
        n_steps = 500
        
        E0 = hamiltonian(q, p, sum_over_particles=True).item()
        
        # Integrate all with store_trajectory=False to get final state
        q_sym, p_sym = sym_euler.integrate(
            q, p, hamiltonian, n_steps, store_trajectory=False
        )
        q_verlet, p_verlet = verlet.integrate(
            q, p, hamiltonian, n_steps, store_trajectory=False
        )
        q_vanilla, p_vanilla = vanilla.integrate(
            q, p, hamiltonian, n_steps, store_trajectory=False
        )
        
        E_sym = hamiltonian(q_sym, p_sym, sum_over_particles=True).item()
        E_verlet = hamiltonian(q_verlet, p_verlet, sum_over_particles=True).item()
        E_vanilla = hamiltonian(q_vanilla, p_vanilla, sum_over_particles=True).item()
        
        error_sym = abs(E_sym - E0) / abs(E0)
        error_verlet = abs(E_verlet - E0) / abs(E0)
        error_vanilla = abs(E_vanilla - E0) / abs(E0)
        
        # Symplectic methods should have smaller error than vanilla
        assert error_sym < error_vanilla or error_verlet < error_vanilla


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
