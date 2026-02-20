"""
Symplectic Integrators for Hamiltonian Systems.

Symplectic integrators preserve the symplectic structure of Hamiltonian systems,
leading to bounded energy error over long time horizons. This is crucial for
orbital mechanics where energy conservation is physically meaningful.

Key properties:
1. Preserve phase space volume (Liouville's theorem)
2. Bounded oscillatory energy error (no secular drift)
3. Exact preservation of quadratic invariants

Implemented integrators:
- Symplectic Euler (1st order)
- Störmer-Verlet / Leapfrog (2nd order)
"""

import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional, List
from abc import ABC, abstractmethod


class BaseIntegrator(ABC, nn.Module):
    """Abstract base class for numerical integrators."""
    
    def __init__(self, dt: float = 1.0, create_graph: bool = False):
        super().__init__()
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float64))
        self.create_graph = create_graph  # For backprop during training
    
    @abstractmethod
    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        hamiltonian: nn.Module,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single integration step.
        
        Args:
            q: Positions of shape (N, 3)
            p: Momenta of shape (N, 3)
            hamiltonian: Hamiltonian module with compute_gradients method
            edge_index: Optional edge indices for graph Hamiltonian
            
        Returns:
            Tuple of (q_new, p_new)
        """
        pass
    
    def integrate(
        self,
        q0: torch.Tensor,
        p0: torch.Tensor,
        hamiltonian: nn.Module,
        n_steps: int,
        edge_index: Optional[torch.Tensor] = None,
        store_trajectory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate the system for multiple steps.
        
        Args:
            q0: Initial positions
            p0: Initial momenta
            hamiltonian: Hamiltonian module
            n_steps: Number of integration steps
            edge_index: Optional edge indices
            store_trajectory: If True, store full trajectory
            
        Returns:
            If store_trajectory: (q_traj, p_traj) of shape (n_steps+1, N, 3)
            Else: (q_final, p_final)
        """
        q, p = q0.clone(), p0.clone()
        
        if store_trajectory:
            q_traj = [q.clone().detach()]
            p_traj = [p.clone().detach()]
        
        for _ in range(n_steps):
            q, p = self.step(q, p, hamiltonian, edge_index)
            
            if store_trajectory:
                q_traj.append(q.clone().detach())
                p_traj.append(p.clone().detach())
        
        if store_trajectory:
            return torch.stack(q_traj), torch.stack(p_traj)
        return q, p


class SymplecticEuler(BaseIntegrator):
    """
    Symplectic Euler integrator (1st order).
    
    Also known as semi-implicit Euler. The key is that we update
    p first, then use the NEW p to update q (or vice versa).
    
    Update equations:
        p_{n+1} = p_n - dt * ∂H/∂q(q_n, p_n)
        q_{n+1} = q_n + dt * ∂H/∂p(q_n, p_{n+1})
    
    This is symplectic because it can be written as the composition
    of two shear maps, each of which is symplectic.
    
    Properties:
    - 1st order accurate
    - Symplectic (preserves phase space structure)
    - Bounded energy oscillations
    """
    
    def __init__(self, dt: float = 1.0, method: str = "pq", create_graph: bool = False):
        """
        Args:
            dt: Time step size
            method: "pq" updates p first, then q; "qp" updates q first
            create_graph: If True, build computation graph for backprop
        """
        super().__init__(dt=dt, create_graph=create_graph)
        self.method = method
    
    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        hamiltonian: nn.Module,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one symplectic Euler step."""
        
        if self.method == "pq":
            # Update p first using current q
            if hasattr(hamiltonian, 'compute_gradients'):
                if edge_index is not None:
                    grad_q, _ = hamiltonian.compute_gradients(q, p, edge_index, create_graph=self.create_graph, dt=self.dt)
                else:
                    grad_q, _ = hamiltonian.compute_gradients(q, p, create_graph=self.create_graph, dt=self.dt)
            else:
                grad_q, _ = self._compute_gradients(q, p, hamiltonian)
            
            p_new = p - self.dt * grad_q
            
            # Update q using NEW p
            if hasattr(hamiltonian, 'compute_gradients'):
                if edge_index is not None:
                    _, grad_p = hamiltonian.compute_gradients(q, p_new, edge_index, create_graph=self.create_graph, dt=self.dt)
                else:
                    _, grad_p = hamiltonian.compute_gradients(q, p_new, create_graph=self.create_graph, dt=self.dt)
            else:
                _, grad_p = self._compute_gradients(q, p_new, hamiltonian)
            
            q_new = q + self.dt * grad_p
            
        else:  # method == "qp"
            # Update q first
            if hasattr(hamiltonian, 'compute_gradients'):
                if edge_index is not None:
                    _, grad_p = hamiltonian.compute_gradients(q, p, edge_index, create_graph=self.create_graph, dt=self.dt)
                else:
                    _, grad_p = hamiltonian.compute_gradients(q, p, create_graph=self.create_graph, dt=self.dt)
            else:
                _, grad_p = self._compute_gradients(q, p, hamiltonian)
            
            q_new = q + self.dt * grad_p
            
            # Update p using NEW q
            if hasattr(hamiltonian, 'compute_gradients'):
                if edge_index is not None:
                    grad_q, _ = hamiltonian.compute_gradients(q_new, p, edge_index, create_graph=self.create_graph, dt=self.dt)
                else:
                    grad_q, _ = hamiltonian.compute_gradients(q_new, p, create_graph=self.create_graph, dt=self.dt)
            else:
                grad_q, _ = self._compute_gradients(q_new, p, hamiltonian)
            
            p_new = p - self.dt * grad_q
        
        return q_new, p_new
    
    def _compute_gradients(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        hamiltonian: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback gradient computation via autograd."""
        with torch.enable_grad():
            q_grad = q.detach().clone().requires_grad_(True)
            p_grad = p.detach().clone().requires_grad_(True)
            
            H = hamiltonian(q_grad, p_grad, sum_over_particles=True)
            
            grad_q, grad_p = torch.autograd.grad(
                H, [q_grad, p_grad],
                create_graph=self.create_graph,
                retain_graph=self.create_graph,
            )
        
        return grad_q, grad_p


class StormerVerlet(BaseIntegrator):
    """
    Störmer-Verlet / Leapfrog integrator (2nd order symplectic).
    
    This is the most commonly used symplectic integrator due to
    its good accuracy-to-cost ratio and time-reversibility.
    
    Update equations (velocity Verlet form):
        p_{n+1/2} = p_n - (dt/2) * ∂H/∂q(q_n)
        q_{n+1} = q_n + dt * ∂H/∂p(p_{n+1/2})
        p_{n+1} = p_{n+1/2} - (dt/2) * ∂H/∂q(q_{n+1})
    
    Properties:
    - 2nd order accurate
    - Symplectic
    - Time-reversible
    - Bounded energy oscillations
    """
    
    def __init__(self, dt: float = 1.0, create_graph: bool = False):
        """
        Args:
            dt: Time step size
            create_graph: If True, build computation graph for backprop
        """
        super().__init__(dt=dt, create_graph=create_graph)
    
    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        hamiltonian: nn.Module,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one Störmer-Verlet step."""
        dt = self.dt
        half_dt = dt / 2
        
        # Half step in p
        if hasattr(hamiltonian, 'compute_gradients'):
            if edge_index is not None:
                grad_q, _ = hamiltonian.compute_gradients(q, p, edge_index, create_graph=self.create_graph, dt=self.dt)
            else:
                grad_q, _ = hamiltonian.compute_gradients(q, p, create_graph=self.create_graph, dt=self.dt)
        else:
            grad_q, _ = self._compute_gradients(q, p, hamiltonian)
        
        p_half = p - half_dt * grad_q
        
        # Full step in q using p_half directly (since dH/dp = p for separable H)
        # For non-separable H, we would need to compute grad_p
        if hasattr(hamiltonian, 'compute_gradients'):
            if edge_index is not None:
                _, grad_p = hamiltonian.compute_gradients(q, p_half, edge_index, create_graph=self.create_graph, dt=self.dt)
            else:
                _, grad_p = hamiltonian.compute_gradients(q, p_half, create_graph=self.create_graph, dt=self.dt)
        else:
            _, grad_p = self._compute_gradients(q, p_half, hamiltonian)
        
        q_new = q + dt * grad_p
        
        # Half step in p with new q
        if hasattr(hamiltonian, 'compute_gradients'):
            if edge_index is not None:
                grad_q_new, _ = hamiltonian.compute_gradients(q_new, p_half, edge_index, create_graph=self.create_graph, dt=self.dt)
            else:
                grad_q_new, _ = hamiltonian.compute_gradients(q_new, p_half, create_graph=self.create_graph, dt=self.dt)
        else:
            grad_q_new, _ = self._compute_gradients(q_new, p_half, hamiltonian)
        
        p_new = p_half - half_dt * grad_q_new
        
        return q_new, p_new
    
    def _compute_gradients(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        hamiltonian: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback gradient computation via autograd."""
        with torch.enable_grad():
            q_grad = q.detach().clone().requires_grad_(True)
            p_grad = p.detach().clone().requires_grad_(True)
            
            H = hamiltonian(q_grad, p_grad, sum_over_particles=True)
            
            grad_q, grad_p = torch.autograd.grad(
                H, [q_grad, p_grad],
                create_graph=self.create_graph,
                retain_graph=self.create_graph,
            )
        
        return grad_q, grad_p


class SymplecticRK(BaseIntegrator):
    """
    4th order symplectic Runge-Kutta integrator.
    
    Uses the Yoshida (1990) decomposition for a 4th order
    symplectic integrator based on second-order building blocks.
    
    This achieves 4th order accuracy while maintaining symplecticity.
    """
    
    # Yoshida coefficients for 4th order
    YOSHIDA_W0 = -2**(1/3) / (2 - 2**(1/3))
    YOSHIDA_W1 = 1 / (2 - 2**(1/3))
    
    def __init__(self, dt: float = 1.0, create_graph: bool = False):
        super().__init__(dt=dt, create_graph=create_graph)
        self.verlet = StormerVerlet(dt=dt, create_graph=create_graph)
        
        # Precompute Yoshida substep sizes
        c1 = self.YOSHIDA_W1 / 2
        c2 = (self.YOSHIDA_W0 + self.YOSHIDA_W1) / 2
        c3 = c2
        c4 = c1
        
        self.c_coeffs = [c1, c2, c3, c4]
        self.d_coeffs = [self.YOSHIDA_W1, self.YOSHIDA_W0, self.YOSHIDA_W1]
    
    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        hamiltonian: nn.Module,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one 4th order symplectic RK step."""
        dt = self.dt
        
        # Yoshida's triple-jump composition
        # Three Verlet steps with scaled dt
        sub_dts = [self.YOSHIDA_W1 * dt, self.YOSHIDA_W0 * dt, self.YOSHIDA_W1 * dt]
        
        for sub_dt in sub_dts:
            self.verlet.dt = sub_dt
            q, p = self.verlet.step(q, p, hamiltonian, edge_index)
        
        # Restore original dt
        self.verlet.dt = dt
        
        return q, p
