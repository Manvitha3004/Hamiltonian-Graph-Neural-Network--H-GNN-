"""
Two-Body and Keplerian Hamiltonian implementations.

The Hamiltonian is the total energy of the system and must be:
1. Scalar-valued (returns single number per system)
2. Invariant under rotations (SO(3) invariant)
3. Differentiable for computing dynamics via autograd

H = T + V = (1/2)|p|² - μ/|q|

where:
- q ∈ ℝ³ is position
- p ∈ ℝ³ is momentum (velocity for unit mass)
- μ is the gravitational parameter (GM)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TwoBodyHamiltonian(nn.Module):
    """
    Two-body Keplerian Hamiltonian for orbital mechanics.
    
    H(q, p) = (1/2)|p|² - μ/|q|
    
    This is the fundamental building block. The Hamiltonian is:
    - Scalar invariant (rotationally symmetric)
    - Conserved along true trajectories
    - Used to derive equations of motion via Hamilton's equations
    
    Args:
        mu: Gravitational parameter (GM). Default is Earth's μ in km³/s².
        softening: Small value to prevent singularity at |q|=0.
    """
    
    def __init__(
        self,
        mu: float = 398600.4418,  # Earth's GM in km³/s²
        softening: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float64))
        self.register_buffer("softening", torch.tensor(softening, dtype=torch.float64))
    
    def kinetic_energy(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute kinetic energy T = (1/2)|p|².
        
        Args:
            p: Momentum tensor of shape (..., 3)
            
        Returns:
            Kinetic energy of shape (...)
        """
        return 0.5 * torch.sum(p ** 2, dim=-1)
    
    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute gravitational potential energy V = -μ/|q|.
        
        Args:
            q: Position tensor of shape (..., 3)
            
        Returns:
            Potential energy of shape (...)
        """
        r = torch.norm(q, dim=-1)
        # Softening prevents division by zero
        return -self.mu / (r + self.softening)
    
    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        sum_over_particles: bool = True,
    ) -> torch.Tensor:
        """
        Compute total Hamiltonian H = T + V.
        
        Args:
            q: Position tensor of shape (N, 3) or (batch, N, 3)
            p: Momentum tensor of shape (N, 3) or (batch, N, 3)
            sum_over_particles: If True, sum over all particles to get scalar.
            
        Returns:
            Total Hamiltonian (scalar if sum_over_particles=True)
        """
        T = self.kinetic_energy(p)
        V = self.potential_energy(q)
        H = T + V
        
        if sum_over_particles:
            # Sum over particle dimension to get total system energy
            return H.sum(dim=-1) if H.dim() > 0 else H
        return H
    
    def compute_gradients(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        create_graph: bool = False,
        **kwargs,  # Accept and ignore extra args like dt
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients ∂H/∂q and ∂H/∂p using autograd.
        
        These are used in Hamilton's equations:
            dq/dt = ∂H/∂p = p
            dp/dt = -∂H/∂q = -μq/|q|³
        
        Args:
            q: Position tensor (requires_grad should be True)
            p: Momentum tensor (requires_grad should be True)
            create_graph: If True, create computation graph for backprop.
                          Set True during training, False during inference.
            
        Returns:
            Tuple of (∂H/∂q, ∂H/∂p)
        """
        # Use enable_grad to ensure gradients work even inside no_grad context
        with torch.enable_grad():
            # Clone and ensure gradients are tracked
            q_grad = q.detach().clone().requires_grad_(True)
            p_grad = p.detach().clone().requires_grad_(True)
            
            H = self.forward(q_grad, p_grad, sum_over_particles=True)
            
            # Compute gradients
            grad_q, grad_p = torch.autograd.grad(
                H,
                [q_grad, p_grad],
                create_graph=create_graph,
                retain_graph=create_graph,
            )
        
        return grad_q, grad_p
    
    def hamiltonian_vector_field(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Hamiltonian vector field (time derivatives).
        
        Hamilton's equations with symplectic structure J:
            dz/dt = J ∇H(z)
        
        Explicitly:
            dq/dt = ∂H/∂p
            dp/dt = -∂H/∂q
        
        Args:
            q: Position tensor
            p: Momentum tensor
            
        Returns:
            Tuple of (dq/dt, dp/dt)
        """
        grad_q, grad_p = self.compute_gradients(q, p)
        
        # Hamilton's equations
        dq_dt = grad_p   # ∂H/∂p = p (for kinetic energy 1/2|p|²)
        dp_dt = -grad_q  # -∂H/∂q = μq/|q|³ (gravitational force)
        
        return dq_dt, dp_dt


class KeplerianHamiltonian(TwoBodyHamiltonian):
    """
    Extended Keplerian Hamiltonian with J2 perturbation.
    
    H = H_kepler + H_J2
    
    where H_J2 accounts for Earth's oblateness.
    This is optional and used for more realistic orbital mechanics.
    
    Args:
        mu: Gravitational parameter.
        J2: J2 coefficient (Earth's oblateness).
        R_eq: Equatorial radius of central body.
        softening: Numerical softening parameter.
    """
    
    def __init__(
        self,
        mu: float = 398600.4418,
        J2: float = 1.08263e-3,
        R_eq: float = 6378.137,  # km
        softening: float = 1e-8,
        include_J2: bool = False,
    ):
        super().__init__(mu=mu, softening=softening)
        self.register_buffer("J2", torch.tensor(J2, dtype=torch.float64))
        self.register_buffer("R_eq", torch.tensor(R_eq, dtype=torch.float64))
        self.include_J2 = include_J2
    
    def J2_potential(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute J2 perturbation potential.
        
        V_J2 = (μ J2 R_eq²)/(2r³) * (3(z/r)² - 1)
        
        Args:
            q: Position tensor of shape (..., 3)
            
        Returns:
            J2 potential energy of shape (...)
        """
        r = torch.norm(q, dim=-1, keepdim=True)
        z = q[..., 2:3]  # z-component
        
        r_safe = r + self.softening
        sin_phi_sq = (z / r_safe) ** 2  # sin²(latitude)
        
        V_J2 = (self.mu * self.J2 * self.R_eq**2) / (2 * r_safe**3)
        V_J2 = V_J2 * (3 * sin_phi_sq - 1)
        
        return V_J2.squeeze(-1)
    
    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        """Compute total potential including J2 if enabled."""
        V = super().potential_energy(q)
        if self.include_J2:
            V = V + self.J2_potential(q)
        return V


class GraphHamiltonian(nn.Module):
    """
    Graph Hamiltonian combining single-body and interaction terms.
    
    H_G = Σᵢ H_orbit(zᵢ) + λ Σ_{(i,j)∈E} Φ_θ(zᵢ, zⱼ)
    
    where:
    - H_orbit is the two-body Hamiltonian for each object
    - Φ_θ is a learned interaction potential
    - λ controls interaction strength
    
    This is the core of the H-GNN architecture.
    
    Args:
        orbit_hamiltonian: Single-body Hamiltonian module.
        interaction_potential: Learned pairwise interaction potential.
        interaction_strength: Coupling constant λ.
    """
    
    def __init__(
        self,
        orbit_hamiltonian: TwoBodyHamiltonian,
        interaction_potential: Optional[nn.Module] = None,
        interaction_strength: float = 0.01,  # Kept for backward compat, ignored if use_dt_scaling=True
        use_dt_scaling: bool = True,  # Use dt² scaling (backward error aligned)
        alpha_init: float = 1e-4,  # Initial value for learnable magnitude
    ):
        super().__init__()
        self.H_orbit = orbit_hamiltonian
        self.Phi = interaction_potential
        self.use_dt_scaling = use_dt_scaling
        self.register_buffer(
            "lambda_", 
            torch.tensor(interaction_strength, dtype=torch.float64)
        )
        # Learnable global magnitude for interaction (α in H = H_base + dt² × α × (1/N) × Σ Φ)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
    
    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute total graph Hamiltonian.
        
        Uses backward-error aligned dt² scaling when use_dt_scaling=True:
            H = H_base + dt² * (1/N) * Σ Φ_tilde(z_i, z_j)
        
        Args:
            q: Positions of shape (N, 3)
            p: Momenta of shape (N, 3)
            edge_index: Edge indices of shape (2, E) for interactions
            dt: Timestep for dt² scaling (required if use_dt_scaling=True)
            
        Returns:
            Total scalar Hamiltonian
        """
        # Single-body orbital energies
        H_total = self.H_orbit(q, p, sum_over_particles=True)
        
        # Add interaction terms if provided
        if self.Phi is not None and edge_index is not None:
            src, dst = edge_index
            
            # Get states for connected pairs
            q_i, q_j = q[src], q[dst]
            p_i, p_j = p[src], p[dst]
            
            # Compute pairwise interaction potentials
            phi_ij = self.Phi(q_i, p_i, q_j, p_j)
            
            # Sum interaction energies (factor of 0.5 for double counting)
            # Use 1/N scaling to keep per-particle force magnitude stable
            n_particles = q.shape[0]
            H_interaction = 0.5 * phi_ij.sum() / n_particles
            
            # Apply scaling: H = H_base + dt² × α × (1/N) × Σ Φ
            if self.use_dt_scaling and dt is not None:
                # Backward-error aligned with learnable α
                # α controls global magnitude, dt² provides correct scaling
                scale = (dt ** 2) * self.alpha
            else:
                # Legacy lambda scaling
                scale = self.lambda_
            
            H_total = H_total + scale * H_interaction
        
        return H_total
    
    def compute_gradients(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        create_graph: bool = False,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ∂H/∂q and ∂H/∂p for the full graph Hamiltonian."""
        with torch.enable_grad():
            q_grad = q.detach().clone().requires_grad_(True)
            p_grad = p.detach().clone().requires_grad_(True)
            
            H = self.forward(q_grad, p_grad, edge_index, dt=dt)
            
            grad_q, grad_p = torch.autograd.grad(
                H,
                [q_grad, p_grad],
                create_graph=create_graph,
                retain_graph=create_graph,
            )
        
        return grad_q, grad_p


def compute_angular_momentum(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Compute angular momentum L = q × p.
    
    For Keplerian orbits, angular momentum is conserved.
    This is used as a validation metric.
    
    Args:
        q: Position tensor of shape (..., 3)
        p: Momentum tensor of shape (..., 3)
        
    Returns:
        Angular momentum vector of shape (..., 3)
    """
    return torch.cross(q, p, dim=-1)


def compute_eccentricity_vector(
    q: torch.Tensor,
    p: torch.Tensor,
    mu: float = 398600.4418,
) -> torch.Tensor:
    """
    Compute Laplace-Runge-Lenz (eccentricity) vector.
    
    e = (v × L)/μ - r̂
    
    This vector is conserved for Keplerian orbits and points
    toward periapsis. Its magnitude equals the eccentricity.
    
    Args:
        q: Position tensor of shape (..., 3)
        p: Momentum tensor of shape (..., 3)
        mu: Gravitational parameter
        
    Returns:
        Eccentricity vector of shape (..., 3)
    """
    L = compute_angular_momentum(q, p)
    r = torch.norm(q, dim=-1, keepdim=True)
    r_hat = q / (r + 1e-8)
    
    e = torch.cross(p, L, dim=-1) / mu - r_hat
    return e
