"""
Graph Construction for Sparse Neighbor Interactions.

Uses FAISS for efficient O(N log N) radius neighbor search.
The graph structure is used ONLY to determine which pairs
interact in the Hamiltonian, NOT for message passing.

Key principle: Graph defines interaction topology, but
dynamics come from ∇H, not from GNN layers.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class SparseGraph:
    """Container for sparse graph data."""
    edge_index: torch.Tensor      # (2, E) edge indices
    edge_attr: Optional[torch.Tensor] = None  # (E, d) edge features
    num_nodes: int = 0
    num_edges: int = 0


class GraphBuilder(nn.Module):
    """
    Build sparse interaction graph from particle positions.
    
    Uses radius-based neighbor search to construct edges.
    FAISS is used for efficient search when available.
    
    The graph is used to define which pairs (i,j) contribute
    to the interaction Hamiltonian Σ Φ(zᵢ, zⱼ).
    
    Args:
        radius: Interaction radius (particles closer than this interact)
        max_neighbors: Maximum neighbors per particle (for efficiency)
        use_faiss: Whether to use FAISS for neighbor search
        self_loops: Whether to include self-interactions
    """
    
    def __init__(
        self,
        radius: float = 1000.0,  # km for orbital mechanics
        max_neighbors: int = 50,
        use_faiss: bool = True,
        self_loops: bool = False,
    ):
        super().__init__()
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.use_faiss = use_faiss
        self.self_loops = self_loops
        
        # Check FAISS availability
        self._faiss_available = False
        if use_faiss:
            try:
                import faiss
                self._faiss_available = True
            except ImportError:
                import warnings
                warnings.warn(
                    "FAISS not installed, falling back to brute force. "
                    "Install with: pip install faiss-cpu"
                )
    
    def forward(
        self,
        positions: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> SparseGraph:
        """
        Build sparse graph from positions.
        
        Args:
            positions: (N, 3) particle positions
            batch: (N,) batch indices for batched graphs
            
        Returns:
            SparseGraph with edge_index
        """
        if self._faiss_available and self.use_faiss:
            return self._build_with_faiss(positions, batch)
        else:
            return self._build_brute_force(positions, batch)
    
    def _build_with_faiss(
        self,
        positions: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> SparseGraph:
        """Build graph using FAISS radius search."""
        import faiss
        
        device = positions.device
        positions_np = positions.detach().cpu().numpy().astype(np.float32)
        N = positions_np.shape[0]
        
        if batch is not None:
            # Handle batched case
            return self._build_batched_faiss(positions, batch)
        
        # Build FAISS index
        d = 3  # Dimension
        index = faiss.IndexFlatL2(d)
        index.add(positions_np)
        
        # Range search (returns all neighbors within radius)
        # Note: FAISS uses squared distances
        lims, D, I = index.range_search(positions_np, self.radius ** 2)
        
        # Convert to edge index
        src_list = []
        dst_list = []
        
        for i in range(N):
            start, end = lims[i], lims[i + 1]
            neighbors = I[start:end]
            
            # Limit neighbors if needed
            if len(neighbors) > self.max_neighbors:
                # Keep closest neighbors
                dists = D[start:end]
                closest_idx = np.argsort(dists)[:self.max_neighbors]
                neighbors = neighbors[closest_idx]
            
            for j in neighbors:
                if not self.self_loops and i == j:
                    continue
                src_list.append(i)
                dst_list.append(j)
        
        if len(src_list) == 0:
            # No edges found
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor(
                [src_list, dst_list],
                dtype=torch.long,
                device=device,
            )
        
        return SparseGraph(
            edge_index=edge_index,
            num_nodes=N,
            num_edges=edge_index.shape[1],
        )
    
    def _build_batched_faiss(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> SparseGraph:
        """Build graph with batched data using FAISS."""
        import faiss
        
        device = positions.device
        unique_batches = batch.unique()
        
        all_src = []
        all_dst = []
        offset = 0
        
        for b in unique_batches:
            mask = batch == b
            pos_b = positions[mask]
            n_b = pos_b.shape[0]
            
            pos_np = pos_b.detach().cpu().numpy().astype(np.float32)
            
            index = faiss.IndexFlatL2(3)
            index.add(pos_np)
            lims, D, I = index.range_search(pos_np, self.radius ** 2)
            
            for i in range(n_b):
                start, end = lims[i], lims[i + 1]
                neighbors = I[start:end]
                
                if len(neighbors) > self.max_neighbors:
                    dists = D[start:end]
                    closest_idx = np.argsort(dists)[:self.max_neighbors]
                    neighbors = neighbors[closest_idx]
                
                for j in neighbors:
                    if not self.self_loops and i == j:
                        continue
                    all_src.append(i + offset)
                    all_dst.append(j + offset)
            
            offset += n_b
        
        if len(all_src) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor(
                [all_src, all_dst],
                dtype=torch.long,
                device=device,
            )
        
        return SparseGraph(
            edge_index=edge_index,
            num_nodes=positions.shape[0],
            num_edges=edge_index.shape[1],
        )
    
    def _build_brute_force(
        self,
        positions: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> SparseGraph:
        """Build graph using brute force pairwise distances."""
        device = positions.device
        N = positions.shape[0]
        
        # Compute all pairwise distances
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
        dist_sq = (diff ** 2).sum(dim=-1)  # (N, N)
        
        # Find edges within radius
        mask = dist_sq < self.radius ** 2
        
        if not self.self_loops:
            mask.fill_diagonal_(False)
        
        # Handle batch constraints
        if batch is not None:
            batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
            mask = mask & batch_mask
        
        # Convert to edge index
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        
        # Limit neighbors per node if needed
        if self.max_neighbors < N:
            edge_index = self._limit_neighbors(edge_index, dist_sq, N)
        
        return SparseGraph(
            edge_index=edge_index,
            num_nodes=N,
            num_edges=edge_index.shape[1],
        )
    
    def _limit_neighbors(
        self,
        edge_index: torch.Tensor,
        dist_sq: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Limit number of neighbors per node to max_neighbors."""
        src, dst = edge_index
        
        new_src = []
        new_dst = []
        
        for i in range(N):
            # Find all neighbors of node i
            mask = src == i
            neighbors = dst[mask]
            
            if len(neighbors) > self.max_neighbors:
                # Keep closest
                dists = dist_sq[i, neighbors]
                _, closest_idx = torch.topk(dists, self.max_neighbors, largest=False)
                neighbors = neighbors[closest_idx]
            
            for j in neighbors:
                new_src.append(i)
                new_dst.append(j.item())
        
        if len(new_src) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
        
        return torch.tensor([new_src, new_dst], dtype=torch.long, device=edge_index.device)
    
    def compute_edge_features(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute invariant edge features for interaction potential.
        
        Features are constructed to be SO(3) invariant:
        - ||Δq|| : relative position magnitude
        - ||Δp|| : relative momentum magnitude
        - Δq · Δp : dot product (invariant scalar)
        - ||Δq||² : squared distance
        - Δq · (pᵢ + pⱼ) : momentum alignment
        
        Args:
            q: Positions (N, 3)
            p: Momenta (N, 3)
            edge_index: (2, E) edge indices
            
        Returns:
            Edge features (E, num_features)
        """
        src, dst = edge_index
        
        # Relative vectors
        delta_q = q[dst] - q[src]  # (E, 3)
        delta_p = p[dst] - p[src]  # (E, 3)
        
        # Average momenta
        p_avg = 0.5 * (p[src] + p[dst])  # (E, 3)
        
        # Invariant features
        r = torch.norm(delta_q, dim=-1, keepdim=True)  # (E, 1)
        r_sq = r ** 2
        delta_p_norm = torch.norm(delta_p, dim=-1, keepdim=True)
        dot_qp = (delta_q * delta_p).sum(dim=-1, keepdim=True)
        dot_q_pavg = (delta_q * p_avg).sum(dim=-1, keepdim=True)
        
        # Combine features
        edge_features = torch.cat([
            r,
            r_sq,
            delta_p_norm,
            dot_qp,
            dot_q_pavg,
        ], dim=-1)
        
        return edge_features


class DynamicGraphBuilder(GraphBuilder):
    """
    Graph builder with dynamic radius adaptation.
    
    Adjusts radius based on local density to maintain
    approximately constant number of neighbors.
    """
    
    def __init__(
        self,
        target_neighbors: int = 20,
        min_radius: float = 100.0,
        max_radius: float = 10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_neighbors = target_neighbors
        self.min_radius = min_radius
        self.max_radius = max_radius
    
    def forward(
        self,
        positions: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> SparseGraph:
        """Build graph with adaptive radius."""
        # Start with base radius and adjust
        graph = super().forward(positions, batch)
        
        # Check average degree
        avg_degree = graph.num_edges / max(graph.num_nodes, 1)
        
        # Adaptive adjustment (simple heuristic)
        if avg_degree < self.target_neighbors * 0.5:
            # Increase radius
            self.radius = min(self.radius * 1.5, self.max_radius)
            graph = super().forward(positions, batch)
        elif avg_degree > self.target_neighbors * 2:
            # Decrease radius
            self.radius = max(self.radius * 0.7, self.min_radius)
            graph = super().forward(positions, batch)
        
        return graph
