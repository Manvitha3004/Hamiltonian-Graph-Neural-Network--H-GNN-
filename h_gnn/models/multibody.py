"""
Multi-Body Gravitational Dynamics for H-GNN Testing.

This creates a ground truth with inter-satellite gravitational interactions
that the H-GNN's pairwise Φ can actually learn.

Unlike J2 (single-body), this is truly pairwise: each satellite
gravitationally attracts every other satellite.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MultiBodyHamiltonian(nn.Module):
    """
    Multi-body gravitational Hamiltonian.
    
    H = Σᵢ H_orbit(zᵢ) + ε Σ_{i<j} m_i m_j / |rᵢ - rⱼ|
    
    where ε controls the strength of inter-satellite gravity.
    
    For debris with typical mass ~1000 kg and separation ~1000 km,
    the real gravitational acceleration is ~10⁻¹² m/s².
    
    We artificially amplify ε to create a learnable signal.
    """
    
    def __init__(
        self,
        mu: float = 398600.4418,
        epsilon: float = 1.0,  # Inter-satellite coupling strength
        satellite_mass: float = 1000.0,  # kg (for scaling)
        softening: float = 10.0,  # km, prevents singularity
    ):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float64))
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float64))
        self.register_buffer("m_sat", torch.tensor(satellite_mass, dtype=torch.float64))
        self.register_buffer("softening", torch.tensor(softening, dtype=torch.float64))
    
    def kinetic_energy(self, p: torch.Tensor) -> torch.Tensor:
        """T = Σᵢ |pᵢ|²/2"""
        return 0.5 * torch.sum(p**2, dim=-1)
    
    def orbital_potential(self, q: torch.Tensor) -> torch.Tensor:
        """V_orbit = -μ/|r| for each satellite"""
        r = torch.norm(q, dim=-1, keepdim=True) + 1e-10
        return -self.mu / r.squeeze(-1)
    
    def interaction_potential(self, q: torch.Tensor) -> torch.Tensor:
        """
        V_int = -ε Σ_{i<j} / |rᵢ - rⱼ|
        
        Returns scalar total interaction energy.
        """
        n = q.shape[0]
        if n < 2:
            return torch.tensor(0.0, dtype=q.dtype, device=q.device)
        
        # Compute all pairwise distances
        # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        diff = q.unsqueeze(1) - q.unsqueeze(0)  # (N, N, 3)
        dist = torch.norm(diff, dim=-1)  # (N, N)
        
        # Add softening to prevent singularity
        dist_safe = dist + self.softening
        
        # Mask diagonal (self-interaction)
        mask = 1 - torch.eye(n, dtype=q.dtype, device=q.device)
        
        # Sum upper triangle (i < j)
        potential = -self.epsilon / dist_safe
        potential = potential * mask
        
        # Sum and divide by 2 (since we counted both (i,j) and (j,i))
        return 0.5 * potential.sum()
    
    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        sum_over_particles: bool = True,
    ) -> torch.Tensor:
        """
        Compute total Hamiltonian H = T + V_orbit + V_int.
        """
        T = self.kinetic_energy(p)  # (N,)
        V_orbit = self.orbital_potential(q)  # (N,)
        V_int = self.interaction_potential(q)  # scalar
        
        if sum_over_particles:
            return T.sum() + V_orbit.sum() + V_int
        return T + V_orbit  # V_int is already summed
    
    def compute_gradients(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,  # Ignored, for API compatibility
        create_graph: bool = False,
        **kwargs,  # Accept and ignore extra args like dt
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ∂H/∂q and ∂H/∂p.
        
        Note: edge_index is ignored since MultiBodyHamiltonian computes
        interactions over all pairs. Included for API compatibility
        with graph-based Hamiltonians.
        """
        with torch.enable_grad():
            q_grad = q.detach().clone().requires_grad_(True)
            p_grad = p.detach().clone().requires_grad_(True)
            
            H = self.forward(q_grad, p_grad, sum_over_particles=True)
            
            grad_q, grad_p = torch.autograd.grad(
                H,
                [q_grad, p_grad],
                create_graph=create_graph,
                retain_graph=create_graph,
            )
        
        return grad_q, grad_p


def generate_multibody_trajectories(
    n_objects: int = 50,
    n_steps: int = 100,
    dt: float = 60.0,
    epsilon: float = 100.0,  # Artificially strong coupling
    mu: float = 398600.4418,
    seed: int = 42,
) -> dict:
    """
    Generate trajectories with multi-body gravitational interactions.
    
    The epsilon parameter controls how strong the inter-satellite
    gravity is. Set high to create a learnable signal.
    """
    from h_gnn.data import SyntheticCatalog
    from h_gnn.integrators import StormerVerlet
    from h_gnn.models.hamiltonian import TwoBodyHamiltonian
    
    # Generate initial conditions
    catalog = SyntheticCatalog(mu=mu, seed=seed)
    q0, p0 = catalog.generate_training_set(n_objects)
    
    # Multi-body Hamiltonian
    multibody_H = MultiBodyHamiltonian(mu=mu, epsilon=epsilon)
    twobody_H = TwoBodyHamiltonian(mu=mu)
    integrator = StormerVerlet(dt=dt)
    
    # Generate ground truth with multi-body interactions
    with torch.no_grad():
        q_mb, p_mb = integrator.integrate(
            q0, p0, multibody_H, n_steps, store_trajectory=False
        )
        q_tb, p_tb = integrator.integrate(
            q0, p0, twobody_H, n_steps, store_trajectory=False
        )
    
    # Compute deviation caused by inter-satellite gravity
    deviation = torch.norm(q_mb - q_tb, dim=-1).mean().item()
    
    return {
        "q_initial": q0,
        "p_initial": p0,
        "q_multibody": q_mb,
        "p_multibody": p_mb,
        "q_twobody": q_tb,
        "p_twobody": p_tb,
        "deviation_km": deviation,
        "epsilon": epsilon,
    }


def test_multibody_signal():
    """Test that multi-body dynamics creates a learnable signal."""
    print("=" * 60)
    print("Multi-Body Interaction Signal Analysis")
    print("=" * 60)
    
    for epsilon in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        result = generate_multibody_trajectories(
            n_objects=50, n_steps=50, epsilon=epsilon
        )
        print(f"ε = {epsilon:6.1f}: deviation = {result['deviation_km']:8.2f} km")


if __name__ == "__main__":
    test_multibody_signal()
