"""
Baseline GNN Models for Comparison.

These are standard GNN architectures that use message passing
for dynamics, rather than Hamiltonian mechanics.

Models:
1. VanillaGNN: Standard message passing GNN
2. EquivariantGNN: SE(3) equivariant GNN (using e3nn when available)

These serve as baselines to demonstrate the benefit of
Hamiltonian structure preservation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import warnings


class VanillaGNN(nn.Module):
    """
    Standard message passing GNN for trajectory prediction.
    
    This baseline learns dynamics through message passing,
    NOT through Hamiltonian mechanics. It serves to show
    that structure preservation matters.
    
    Architecture:
        Node features: [q, p] ∈ ℝ⁶
        Message: MLP(concat(node_i, node_j, edge_ij))
        Update: MLP(node + aggregate(messages))
        
    The model directly predicts (Δq, Δp) for one step.
    
    Args:
        hidden_dim: Hidden dimension for MLPs
        n_layers: Number of message passing layers
        aggregation: Aggregation method ("sum", "mean", "max")
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 3,
        aggregation: str = "sum",
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.aggregation = aggregation
        
        # Input embedding: [q, p] -> hidden
        self.node_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Message passing layers
        self.message_mlps = nn.ModuleList()
        self.update_mlps = nn.ModuleList()
        
        for _ in range(n_layers):
            # Message MLP: concat(h_i, h_j, edge_feat) -> message
            self.message_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim * 2 + 5, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            
            # Update MLP: h + aggregated_msg -> new_h
            self.update_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        
        # Output decoder: hidden -> [Δq, Δp]
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )
    
    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: torch.Tensor,
        n_steps: int = 1,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass predicting trajectory.
        
        Args:
            q: Positions (N, 3)
            p: Momenta (N, 3)
            edge_index: Edge indices (2, E)
            n_steps: Number of prediction steps
            return_trajectory: Return full trajectory
            
        Returns:
            Predicted (q, p) or trajectory
        """
        if return_trajectory:
            q_traj = [q.clone()]
            p_traj = [p.clone()]
        
        q_curr, p_curr = q.clone(), p.clone()
        
        for _ in range(n_steps):
            delta_q, delta_p = self._step(q_curr, p_curr, edge_index)
            q_curr = q_curr + delta_q
            p_curr = p_curr + delta_p
            
            if return_trajectory:
                q_traj.append(q_curr.clone())
                p_traj.append(p_curr.clone())
        
        if return_trajectory:
            return torch.stack(q_traj), torch.stack(p_traj)
        return q_curr, p_curr
    
    def _step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step prediction."""
        N = q.shape[0]
        src, dst = edge_index
        
        # Initial node features
        x = torch.cat([q, p], dim=-1)  # (N, 6)
        h = self.node_encoder(x)  # (N, hidden)
        
        # Compute edge features (invariant)
        delta_q = q[dst] - q[src]
        delta_p = p[dst] - p[src]
        r = torch.norm(delta_q, dim=-1, keepdim=True)
        r_sq = r ** 2
        delta_p_norm = torch.norm(delta_p, dim=-1, keepdim=True)
        dot_qp = (delta_q * delta_p).sum(dim=-1, keepdim=True)
        p_avg = 0.5 * (p[src] + p[dst])
        dot_q_pavg = (delta_q * p_avg).sum(dim=-1, keepdim=True)
        
        edge_feat = torch.cat([r, r_sq, delta_p_norm, dot_qp, dot_q_pavg], dim=-1)
        
        # Message passing
        for msg_mlp, upd_mlp in zip(self.message_mlps, self.update_mlps):
            # Compute messages
            msg_input = torch.cat([h[src], h[dst], edge_feat], dim=-1)
            messages = msg_mlp(msg_input)  # (E, hidden)
            
            # Aggregate messages
            agg = torch.zeros(N, self.hidden_dim, device=q.device)
            if self.aggregation == "sum":
                agg.index_add_(0, dst, messages)
            elif self.aggregation == "mean":
                counts = torch.zeros(N, device=q.device)
                agg.index_add_(0, dst, messages)
                counts.index_add_(0, dst, torch.ones(len(dst), device=q.device))
                agg = agg / (counts.unsqueeze(-1) + 1e-8)
            else:  # max
                for i in range(len(dst)):
                    agg[dst[i]] = torch.max(agg[dst[i]], messages[i])
            
            # Update
            h = upd_mlp(torch.cat([h, agg], dim=-1))
        
        # Decode to deltas
        output = self.decoder(h)  # (N, 6)
        delta_q = output[:, :3]
        delta_p = output[:, 3:]
        
        return delta_q, delta_p


class EquivariantGNN(nn.Module):
    """
    SE(3)-Equivariant GNN baseline.
    
    Uses e3nn for equivariant message passing. The output
    transformation is equivariant: if input is rotated,
    output rotates accordingly.
    
    Note: This does NOT preserve Hamiltonian structure.
    It's equivariant but not symplectic.
    
    Requires e3nn: pip install e3nn
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        lmax: int = 2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lmax = lmax
        
        # Check if e3nn is available
        try:
            import e3nn
            from e3nn import o3
            from e3nn.nn import FullyConnectedNet, Gate
            self._e3nn_available = True
        except ImportError:
            warnings.warn(
                "e3nn not installed. EquivariantGNN will use fallback. "
                "Install with: pip install e3nn"
            )
            self._e3nn_available = False
            return
        
        # Define irreps
        # Input: scalar features (l=0) + vector positions/momenta (l=1)
        self.irreps_input = o3.Irreps("2x0e + 2x1o")  # 2 scalars + 2 vectors
        self.irreps_hidden = o3.Irreps(f"{hidden_dim}x0e + {hidden_dim//2}x1o")
        self.irreps_output = o3.Irreps("1x1o + 1x1o")  # Δq and Δp (vectors)
        
        # Edge spherical harmonics
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        
        # Build layers using e3nn convolutions
        from e3nn.nn.models.gate_points_2101 import SimpleNetwork
        
        # Simplified: use e3nn's SimpleNetwork
        self.network = SimpleNetwork(
            irreps_in=self.irreps_input,
            irreps_hidden=self.irreps_hidden,
            irreps_out=self.irreps_output,
            irreps_node_attr="0e",
            irreps_edge_attr=self.irreps_sh,
            layers=n_layers,
            max_radius=1000.0,
            number_of_basis=8,
            radial_layers=2,
            radial_neurons=hidden_dim,
            num_neighbors=20,
        )
    
    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: torch.Tensor,
        n_steps: int = 1,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with equivariant dynamics."""
        
        if not self._e3nn_available:
            # Fallback to vanilla GNN behavior
            return self._fallback_forward(q, p, edge_index, n_steps, return_trajectory)
        
        if return_trajectory:
            q_traj = [q.clone()]
            p_traj = [p.clone()]
        
        q_curr, p_curr = q.clone(), p.clone()
        
        for _ in range(n_steps):
            delta_q, delta_p = self._equivariant_step(q_curr, p_curr, edge_index)
            q_curr = q_curr + delta_q
            p_curr = p_curr + delta_p
            
            if return_trajectory:
                q_traj.append(q_curr.clone())
                p_traj.append(p_curr.clone())
        
        if return_trajectory:
            return torch.stack(q_traj), torch.stack(p_traj)
        return q_curr, p_curr
    
    def _equivariant_step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single equivariant step."""
        from e3nn import o3
        
        N = q.shape[0]
        
        # Build node features in irreps format
        # Scalars: |q|, |p|
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        p_norm = torch.norm(p, dim=-1, keepdim=True)
        
        # Vectors: q, p (normalized for numerical stability)
        q_vec = q / (q_norm + 1e-8)
        p_vec = p / (p_norm + 1e-8)
        
        # Concatenate: [scalar_q, scalar_p, vec_q, vec_p]
        node_features = torch.cat([q_norm, p_norm, q_vec, p_vec], dim=-1)
        
        # Node attributes (optional, using constant)
        node_attr = torch.ones(N, 1, device=q.device)
        
        # Edge vectors
        src, dst = edge_index
        edge_vec = q[dst] - q[src]
        
        # Run network
        output = self.network(
            node_features,
            q,  # positions
            edge_index,
            node_attr,
        )
        
        # Parse output into Δq and Δp
        delta_q = output[:, :3]
        delta_p = output[:, 3:6]
        
        return delta_q, delta_p
    
    def _fallback_forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: torch.Tensor,
        n_steps: int,
        return_trajectory: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback when e3nn not available."""
        # Simple linear prediction
        if return_trajectory:
            q_traj = [q.clone()]
            p_traj = [p.clone()]
        
        q_curr, p_curr = q.clone(), p.clone()
        
        for _ in range(n_steps):
            # Simple physics: dq/dt ≈ p, dp/dt ≈ -μq/|q|³
            dt = 60.0  # Assume 1 minute step
            mu = 398600.4418
            
            r = torch.norm(q_curr, dim=-1, keepdim=True) + 1e-8
            accel = -mu * q_curr / (r ** 3)
            
            q_curr = q_curr + dt * p_curr
            p_curr = p_curr + dt * accel
            
            if return_trajectory:
                q_traj.append(q_curr.clone())
                p_traj.append(p_curr.clone())
        
        if return_trajectory:
            return torch.stack(q_traj), torch.stack(p_traj)
        return q_curr, p_curr


class RecurrentGNN(nn.Module):
    """
    GNN with recurrent state for sequence modeling.
    
    Uses GRU cells to maintain temporal state across
    prediction steps. Does NOT preserve Hamiltonian structure.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_gnn_layers: int = 2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # GNN encoder
        self.gnn = VanillaGNN(hidden_dim=hidden_dim, n_layers=n_gnn_layers)
        
        # GRU for temporal modeling
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # State encoder/decoder
        self.state_encoder = nn.Linear(6, hidden_dim)
        self.state_decoder = nn.Linear(hidden_dim, 6)
    
    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: torch.Tensor,
        n_steps: int = 1,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with recurrent state."""
        N = q.shape[0]
        
        if return_trajectory:
            q_traj = [q.clone()]
            p_traj = [p.clone()]
        
        # Initialize hidden state
        state = torch.cat([q, p], dim=-1)
        h = self.state_encoder(state)  # (N, hidden)
        
        q_curr, p_curr = q.clone(), p.clone()
        
        for _ in range(n_steps):
            # Get GNN features
            delta_q, delta_p = self.gnn._step(q_curr, p_curr, edge_index)
            gnn_feat = torch.cat([delta_q, delta_p], dim=-1)
            gnn_feat = self.state_encoder(gnn_feat)
            
            # GRU update
            h = self.gru(gnn_feat, h)
            
            # Decode to deltas
            output = self.state_decoder(h)
            delta_q = output[:, :3]
            delta_p = output[:, 3:]
            
            q_curr = q_curr + delta_q
            p_curr = p_curr + delta_p
            
            if return_trajectory:
                q_traj.append(q_curr.clone())
                p_traj.append(p_curr.clone())
        
        if return_trajectory:
            return torch.stack(q_traj), torch.stack(p_traj)
        return q_curr, p_curr


def create_baseline(name: str, **kwargs) -> nn.Module:
    """
    Factory function to create baseline models.
    
    Args:
        name: "vanilla_gnn", "equivariant_gnn", "recurrent_gnn"
        **kwargs: Arguments for the model
        
    Returns:
        Baseline model instance
    """
    baselines = {
        "vanilla_gnn": VanillaGNN,
        "equivariant_gnn": EquivariantGNN,
        "recurrent_gnn": RecurrentGNN,
    }
    
    if name not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(baselines.keys())}")
    
    return baselines[name](**kwargs)
