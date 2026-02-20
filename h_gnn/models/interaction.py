"""
Learned Interaction Potential for Graph Hamiltonian.

The interaction potential Φ(zᵢ, zⱼ) must be:
1. Scalar-valued (outputs single number per pair)
2. Symmetric: Φ(zᵢ, zⱼ) = Φ(zⱼ, zᵢ)
3. SO(3) invariant (for equivariance of dynamics)

Inputs are constructed from invariant quantities:
- ||Δq|| : separation distance
- ||Δp|| : relative momentum magnitude
- Δq · Δp : alignment of separation and relative momentum
"""

import torch
import torch.nn as nn
from typing import Optional, List


class InteractionPotential(nn.Module):
    """
    Learned pairwise interaction potential Φ_θ(zᵢ, zⱼ).
    
    This is a scalar-valued function that contributes to the
    total Hamiltonian: H = Σ H_orbit + λ Σ Φ(zᵢ, zⱼ)
    
    The potential uses invariant features to ensure SO(3)
    invariance of the Hamiltonian.
    
    Architecture:
        Invariant features → MLP → scalar output
    
    Args:
        hidden_dims: List of hidden layer dimensions
        activation: Activation function
        use_residual: Whether to use residual connections
        output_scale: Scale factor for output (helps with stability)
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [64, 64, 32],
        activation: str = "silu",
        use_residual: bool = True,
        output_scale: float = 1.0,
        softplus_output: bool = True,
        normalize_features: bool = False,
        feature_scales: Optional[List[float]] = None,
        init_gain: float = 0.1,  # Control output magnitude at init
    ):
        super().__init__()
        
        self.use_residual = use_residual
        self.output_scale = output_scale
        self.softplus_output = softplus_output
        self.normalize_features = normalize_features
        self.init_gain = init_gain
        
        # Feature scales for normalization: [r, r², ||Δp||, Δq·Δp, Δq·p_avg]
        # Typical magnitudes for LEO: r~100-1000km, p~7km/s  
        if feature_scales is None:
            feature_scales = [1000.0, 1e6, 10.0, 1e4, 1e4]
        self.register_buffer("feature_scales", torch.tensor(feature_scales, dtype=torch.float32))
        
        # Input features: [r, r², ||Δp||, Δq·Δp, Δq·p_avg]
        input_dim = 5
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i, hdim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hdim))
            
            if activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.SiLU())
            
            prev_dim = hdim
        
        # Final layer to scalar
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Optional residual: direct distance-based term
        if use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # Initialize for small initial outputs
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training.
        
        Uses small initialization for final layer so model starts near base Hamiltonian.
        Only zeros the BIAS (not weights) to allow gradient flow.
        """
        layers = [m for m in self.mlp.modules() if isinstance(m, nn.Linear)]
        
        # Initialize all layers including final with xavier
        for m in layers:
            nn.init.xavier_normal_(m.weight, gain=self.init_gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        # Final layer: use very small gain to start near zero output
        # but NOT exactly zero (to allow gradient flow)
        final_layer = layers[-1]
        nn.init.xavier_normal_(final_layer.weight, gain=0.01)
    
    def forward(
        self,
        q_i: torch.Tensor,
        p_i: torch.Tensor,
        q_j: torch.Tensor,
        p_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute interaction potential for pairs.
        
        The function is symmetric by construction:
        We compute features that are invariant to swapping (i, j).
        
        Args:
            q_i: Positions of source nodes (E, 3)
            p_i: Momenta of source nodes (E, 3)
            q_j: Positions of target nodes (E, 3)
            p_j: Momenta of target nodes (E, 3)
            
        Returns:
            Interaction potential values (E,)
        """
        # Relative vectors
        delta_q = q_j - q_i  # (E, 3)
        delta_p = p_j - p_i  # (E, 3)
        
        # Average momentum (symmetric)
        p_avg = 0.5 * (p_i + p_j)  # (E, 3)
        
        # Invariant features
        r = torch.norm(delta_q, dim=-1, keepdim=True) + 1e-8  # (E, 1)
        r_sq = r ** 2
        delta_p_norm = torch.norm(delta_p, dim=-1, keepdim=True)
        
        # Dot products (symmetric under swap)
        dot_qp = (delta_q * delta_p).sum(dim=-1, keepdim=True)
        dot_q_pavg = (delta_q * p_avg).sum(dim=-1, keepdim=True)
        
        # Stack features
        features = torch.cat([
            r,
            r_sq,
            delta_p_norm,
            dot_qp,
            dot_q_pavg,
        ], dim=-1)  # (E, 5)
        
        # Apply feature normalization if enabled
        if self.normalize_features:
            features = features / self.feature_scales
        
        # MLP forward
        phi = self.mlp(features).squeeze(-1)  # (E,)
        
        # Optional residual (simple inverse distance term)
        if self.use_residual:
            phi = phi + self.residual_weight / r.squeeze(-1)
        
        # Ensure non-negative potential (optional, for stability)
        if self.softplus_output:
            phi = torch.nn.functional.softplus(phi)
        
        return self.output_scale * phi


class SymmetricInteractionPotential(nn.Module):
    """
    Explicitly symmetric interaction potential.
    
    Enforces Φ(zᵢ, zⱼ) = Φ(zⱼ, zᵢ) by averaging:
        Φ_sym = 0.5 * (Φ(zᵢ, zⱼ) + Φ(zⱼ, zᵢ))
    
    This is more expensive but guarantees exact symmetry.
    """
    
    def __init__(self, base_potential: InteractionPotential):
        super().__init__()
        self.base = base_potential
    
    def forward(
        self,
        q_i: torch.Tensor,
        p_i: torch.Tensor,
        q_j: torch.Tensor,
        p_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetrized interaction potential."""
        phi_ij = self.base(q_i, p_i, q_j, p_j)
        phi_ji = self.base(q_j, p_j, q_i, p_i)
        return 0.5 * (phi_ij + phi_ji)


class DeepSetInteraction(nn.Module):
    """
    Deep Sets style interaction for permutation invariance.
    
    Uses the structure:
        ρ(Σ φ(zᵢ, zⱼ))
    
    where φ encodes individual interactions and ρ aggregates.
    This naturally handles variable numbers of neighbors.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        output_dim: int = 1,
    ):
        super().__init__()
        
        # Per-pair encoder
        self.phi = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Aggregation network
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        q_i: torch.Tensor,
        p_i: torch.Tensor,
        q_j: torch.Tensor,
        p_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Deep Sets interaction."""
        # Build features
        delta_q = q_j - q_i
        delta_p = p_j - p_i
        p_avg = 0.5 * (p_i + p_j)
        
        r = torch.norm(delta_q, dim=-1, keepdim=True) + 1e-8
        r_sq = r ** 2
        delta_p_norm = torch.norm(delta_p, dim=-1, keepdim=True)
        dot_qp = (delta_q * delta_p).sum(dim=-1, keepdim=True)
        dot_q_pavg = (delta_q * p_avg).sum(dim=-1, keepdim=True)
        
        features = torch.cat([r, r_sq, delta_p_norm, dot_qp, dot_q_pavg], dim=-1)
        
        # Encode
        encoded = self.phi(features)  # (E, hidden_dim)
        
        # Final output (per-edge)
        output = self.rho(encoded).squeeze(-1)  # (E,)
        
        return output


class ScalableInteraction(nn.Module):
    """
    Interaction potential with distance-based cutoff.
    
    Uses smooth cutoff function to ensure interactions
    decay to zero at the cutoff radius, avoiding
    discontinuities in the Hamiltonian.
    
    Args:
        cutoff_radius: Distance beyond which interactions are zero
        hidden_dims: MLP hidden layer dimensions
        softplus_output: If False, allow signed outputs (for learning corrections)
        normalize_features: If True, normalize input features for better learning
    """
    
    def __init__(
        self,
        cutoff_radius: float = 1000.0,
        hidden_dims: List[int] = [64, 64],
        softplus_output: bool = True,
        normalize_features: bool = False,
        feature_scales: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        
        self.base_potential = InteractionPotential(
            hidden_dims=hidden_dims,
            softplus_output=softplus_output,
            normalize_features=normalize_features,
            feature_scales=feature_scales,
            **kwargs,
        )
    
    def cutoff_fn(self, r: torch.Tensor) -> torch.Tensor:
        """
        Smooth cutoff function.
        
        Uses cosine cutoff: 0.5 * (1 + cos(π * r / r_cut)) for r < r_cut
        """
        # Normalized distance
        x = r / self.cutoff_radius
        
        # Cosine cutoff
        cutoff = torch.where(
            x < 1.0,
            0.5 * (1 + torch.cos(torch.pi * x)),
            torch.zeros_like(x),
        )
        
        return cutoff
    
    def forward(
        self,
        q_i: torch.Tensor,
        p_i: torch.Tensor,
        q_j: torch.Tensor,
        p_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cutoff-modulated interaction potential."""
        # Base potential
        phi = self.base_potential(q_i, p_i, q_j, p_j)
        
        # Distance
        r = torch.norm(q_j - q_i, dim=-1) + 1e-8
        
        # Apply cutoff
        phi = phi * self.cutoff_fn(r)
        
        return phi
