"""
Hamiltonian Graph Neural Network (H-GNN) Core Module.

The H-GNN learns dynamics through Hamiltonian mechanics:
1. Define graph-based Hamiltonian H = Σ H_orbit + λ Σ Φ_θ
2. Compute dynamics via ∇H (NOT message passing)
3. Integrate using symplectic integrator

Key principle: The GNN structure is only used for defining
which pairs interact in the Hamiltonian. Dynamics come
EXCLUSIVELY from Hamilton's equations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .hamiltonian import TwoBodyHamiltonian, GraphHamiltonian
from .interaction import InteractionPotential, ScalableInteraction
from .graph import GraphBuilder, SparseGraph


class HGNN(nn.Module):
    """
    Hamiltonian Graph Neural Network.
    
    Complete model that:
    1. Builds sparse interaction graph from positions
    2. Computes graph Hamiltonian (orbital + interaction)
    3. Integrates dynamics using symplectic integrator
    4. Predicts future states or derived quantities
    
    Architecture:
        Input: (q, p) ∈ ℝ^{N×3} × ℝ^{N×3}
        ↓
        Graph Construction (FAISS radius search)
        ↓
        H = Σ H_orbit(qᵢ, pᵢ) + λ Σ Φ_θ(zᵢ, zⱼ)
        ↓
        Symplectic Integration (∇H → dynamics)
        ↓
        Output: Future states or predictions
    
    Args:
        mu: Gravitational parameter
        interaction_strength: λ coupling constant
        interaction_radius: Radius for neighbor search
        hidden_dims: Hidden dimensions for interaction MLP
        integrator_type: "symplectic_euler" or "stormer_verlet"
        dt: Integration time step
        device: Torch device
    """
    
    def __init__(
        self,
        mu: float = 398600.4418,
        interaction_strength: float = 0.01,
        interaction_radius: float = 50000.0,  # 50,000 km covers LEO to GEO
        hidden_dims: list = [64, 64, 32],
        integrator_type: str = "symplectic_euler",
        dt: float = 60.0,  # seconds
        use_cutoff: bool = True,
        max_neighbors: int = 50,
        device: str = "cpu",
        create_graph: bool = False,  # For backprop through integration
        softplus_output: bool = False,  # Allow signed corrections for Hamiltonian learning
        normalize_features: bool = True,  # Normalize MLP input features for stability
        init_gain: float = 0.1,  # Control initial Φ output magnitude
        use_dt_scaling: bool = False,  # dt²-structured correction (backward-error aligned)
        alpha_init: float = 0.5,  # Raw value (clamped: tanh(0.5)*0.01 ≈ 0.0046)
    ):
        super().__init__()
        
        self.dt = dt
        self.device = device
        self.interaction_strength = interaction_strength
        self.alpha_init = alpha_init
        self.create_graph = create_graph
        self.use_dt_scaling = use_dt_scaling
        
        # Two-body orbital Hamiltonian
        self.H_orbit = TwoBodyHamiltonian(mu=mu)
        
        # Learned interaction potential
        if use_cutoff:
            self.interaction = ScalableInteraction(
                cutoff_radius=interaction_radius,
                hidden_dims=hidden_dims,
                softplus_output=softplus_output,
                normalize_features=normalize_features,
                init_gain=init_gain,
            )
        else:
            self.interaction = InteractionPotential(
                hidden_dims=hidden_dims,
                softplus_output=softplus_output,
                normalize_features=normalize_features,  # FIX: was missing!
                init_gain=init_gain,
            )
        
        # Graph construction
        self.graph_builder = GraphBuilder(
            radius=interaction_radius,
            max_neighbors=max_neighbors,
            use_faiss=True,
        )
        
        # Full graph Hamiltonian
        self.hamiltonian = GraphHamiltonian(
            orbit_hamiltonian=self.H_orbit,
            interaction_potential=self.interaction,
            interaction_strength=interaction_strength,
            use_dt_scaling=use_dt_scaling,
            alpha_init=alpha_init,
        )
        
        # Integrator (lazy initialization)
        self.integrator_type = integrator_type
        self._integrator = None
    
    @property
    def integrator(self):
        """Lazy load integrator to avoid circular imports."""
        if self._integrator is None:
            from ..integrators import SymplecticEuler, StormerVerlet
            
            if self.integrator_type == "symplectic_euler":
                self._integrator = SymplecticEuler(dt=self.dt, create_graph=self.create_graph)
            elif self.integrator_type == "stormer_verlet":
                self._integrator = StormerVerlet(dt=self.dt, create_graph=self.create_graph)
            else:
                self._integrator = SymplecticEuler(dt=self.dt, create_graph=self.create_graph)
        
        return self._integrator
    
    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n_steps: int = 1,
        return_trajectory: bool = False,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: integrate system for n_steps.
        
        Args:
            q: Initial positions (N, 3)
            p: Initial momenta (N, 3)
            n_steps: Number of integration steps
            return_trajectory: If True, return full trajectory
            batch: Optional batch indices for multiple systems
            
        Returns:
            Final (q, p) or trajectory if return_trajectory=True
        """
        # Build interaction graph
        graph = self.graph_builder(q, batch)
        edge_index = graph.edge_index
        
        # Integrate
        if return_trajectory:
            q_traj, p_traj = self.integrator.integrate(
                q, p, self.hamiltonian, n_steps,
                edge_index=edge_index,
                store_trajectory=True,
            )
            return q_traj, p_traj
        else:
            q_final, p_final = q.clone(), p.clone()
            for _ in range(n_steps):
                q_final, p_final = self.integrator.step(
                    q_final, p_final, self.hamiltonian, edge_index
                )
            return q_final, p_final
    
    def compute_hamiltonian(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total Hamiltonian value."""
        graph = self.graph_builder(q, batch)
        return self.hamiltonian(q, p, graph.edge_index)
    
    def predict_conjunction(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n_steps: int,
        threshold_distance: float = 10.0,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict minimum separation and conjunction events.
        
        Args:
            q: Initial positions (N, 3)
            p: Initial momenta (N, 3)
            n_steps: Prediction horizon
            threshold_distance: Distance threshold for conjunction
            batch: Batch indices
            
        Returns:
            Dict with min_separation, conjunction_mask, time_to_min
        """
        # Integrate trajectory
        q_traj, p_traj = self.forward(
            q, p, n_steps, return_trajectory=True, batch=batch
        )
        
        N = q.shape[0]
        T = q_traj.shape[0]
        
        # Compute all pairwise distances at each timestep
        min_separations = torch.full((N, N), float('inf'), device=q.device)
        time_to_min = torch.zeros((N, N), dtype=torch.long, device=q.device)
        
        for t in range(T):
            q_t = q_traj[t]
            # Pairwise distances
            diff = q_t.unsqueeze(0) - q_t.unsqueeze(1)  # (N, N, 3)
            dist = torch.norm(diff, dim=-1)  # (N, N)
            
            # Update minimum
            closer = dist < min_separations
            min_separations = torch.where(closer, dist, min_separations)
            time_to_min = torch.where(closer, torch.full_like(time_to_min, t), time_to_min)
        
        # Conjunction detection
        conjunction_mask = min_separations < threshold_distance
        
        return {
            "min_separation": min_separations,
            "conjunction_mask": conjunction_mask,
            "time_to_min": time_to_min,
            "final_q": q_traj[-1],
            "final_p": p_traj[-1],
        }
    
    def get_ablation_config(self) -> Dict[str, bool]:
        """Return current ablation configuration."""
        return {
            "has_interaction": self.interaction is not None,
            "interaction_strength": self.interaction_strength,
            "integrator": self.integrator_type,
        }
    
    def disable_interaction(self):
        """Disable learned interaction (λ=0 ablation)."""
        self.hamiltonian.lambda_ = torch.tensor(0.0)
    
    def enable_interaction(self, strength: Optional[float] = None):
        """Re-enable learned interaction."""
        if strength is not None:
            self.hamiltonian.lambda_ = torch.tensor(strength)
        else:
            self.hamiltonian.lambda_ = torch.tensor(self.interaction_strength)


class HGNNPredictor(nn.Module):
    """
    H-GNN with prediction head for conjunction/separation tasks.
    
    Extends HGNN with a prediction network that takes the
    integrated state and outputs task-specific predictions.
    
    Args:
        hgnn_config: Configuration dict for HGNN
        output_dim: Dimension of prediction output
        prediction_type: "regression" or "classification"
    """
    
    def __init__(
        self,
        hgnn_config: Dict[str, Any],
        output_dim: int = 1,
        prediction_type: str = "regression",
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.hgnn = HGNN(**hgnn_config)
        self.prediction_type = prediction_type
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(6, hidden_dim),  # q, p concatenated
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        if prediction_type == "classification":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
    
    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n_steps: int,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with prediction.
        
        Args:
            q: Initial positions
            p: Initial momenta
            n_steps: Integration steps
            batch: Batch indices
            
        Returns:
            Predictions tensor
        """
        # Integrate
        q_final, p_final = self.hgnn(q, p, n_steps, batch=batch)
        
        # Concatenate for prediction
        state = torch.cat([q_final, p_final], dim=-1)  # (N, 6)
        
        # Predict
        pred = self.predictor(state)
        pred = self.output_activation(pred)
        
        return pred


class PureHamiltonianModel(nn.Module):
    """
    Pure two-body Hamiltonian model (no learned interactions).
    
    This is the λ=0 baseline for ablation studies.
    Tests whether the two-body physics alone is sufficient.
    """
    
    def __init__(
        self,
        mu: float = 398600.4418,
        integrator_type: str = "symplectic_euler",
        dt: float = 60.0,
    ):
        super().__init__()
        
        self.hamiltonian = TwoBodyHamiltonian(mu=mu)
        self.dt = dt
        self.integrator_type = integrator_type
        self._integrator = None
    
    @property
    def integrator(self):
        if self._integrator is None:
            from ..integrators import SymplecticEuler, StormerVerlet
            
            if self.integrator_type == "symplectic_euler":
                self._integrator = SymplecticEuler(dt=self.dt)
            else:
                self._integrator = StormerVerlet(dt=self.dt)
        
        return self._integrator
    
    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n_steps: int = 1,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate pure two-body dynamics."""
        if return_trajectory:
            return self.integrator.integrate(
                q, p, self.hamiltonian, n_steps, store_trajectory=True
            )
        else:
            q_out, p_out = q.clone(), p.clone()
            for _ in range(n_steps):
                q_out, p_out = self.integrator.step(q_out, p_out, self.hamiltonian)
            return q_out, p_out
