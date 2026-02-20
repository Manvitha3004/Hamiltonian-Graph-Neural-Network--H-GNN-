"""
Baseline (Non-Symplectic) Integrators.

These integrators do NOT preserve symplectic structure and will
exhibit energy drift over long integration horizons. They are
implemented for comparison with symplectic methods.

Implemented:
- Vanilla (Forward) Euler: 1st order, non-symplectic
- RK4: 4th order, non-symplectic

Expected behavior:
- Vanilla Euler: Linear energy drift, system diverges
- RK4: Slow energy drift, better than Euler but still drifts
- Symplectic: Bounded oscillatory energy (no drift)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .symplectic import BaseIntegrator


class VanillaEuler(BaseIntegrator):
    """
    Forward (Explicit) Euler integrator.
    
    Update equations:
        q_{n+1} = q_n + dt * ∂H/∂p(q_n, p_n)
        p_{n+1} = p_n - dt * ∂H/∂q(q_n, p_n)
    
    Note: Both updates use the OLD state (q_n, p_n).
    This makes it NOT symplectic.
    
    Properties:
    - 1st order accurate
    - NOT symplectic
    - Energy drifts linearly over time
    - Simple but unsuitable for long-horizon integration
    """
    
    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        hamiltonian: nn.Module,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one forward Euler step (non-symplectic)."""
        
        # Compute gradients at current state
        q_grad = q.clone().requires_grad_(True)
        p_grad = p.clone().requires_grad_(True)
        
        if hasattr(hamiltonian, 'forward'):
            if edge_index is not None and hasattr(hamiltonian, 'compute_gradients'):
                grad_q, grad_p = hamiltonian.compute_gradients(q_grad, p_grad, edge_index)
            elif hasattr(hamiltonian, 'compute_gradients'):
                grad_q, grad_p = hamiltonian.compute_gradients(q_grad, p_grad)
            else:
                H = hamiltonian(q_grad, p_grad, sum_over_particles=True)
                grad_q, grad_p = torch.autograd.grad(
                    H, [q_grad, p_grad],
                    create_graph=False,
                )
        else:
            raise ValueError("Hamiltonian must have a forward method")
        
        # Forward Euler update (both use OLD state - NOT symplectic)
        q_new = q + self.dt * grad_p.detach()
        p_new = p - self.dt * grad_q.detach()
        
        return q_new, p_new


class RK4(BaseIntegrator):
    """
    Classical 4th-order Runge-Kutta integrator.
    
    RK4 is highly accurate but NOT symplectic. For Hamiltonian
    systems, it will exhibit slow energy drift over long horizons.
    
    Update:
        k1 = f(t_n, y_n)
        k2 = f(t_n + dt/2, y_n + dt*k1/2)
        k3 = f(t_n + dt/2, y_n + dt*k2/2)
        k4 = f(t_n + dt, y_n + dt*k3)
        y_{n+1} = y_n + (dt/6)(k1 + 2*k2 + 2*k3 + k4)
    
    Properties:
    - 4th order accurate
    - NOT symplectic
    - Energy drift is slow but present
    - Good for short-term accuracy, poor for long-term structure
    """
    
    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        hamiltonian: nn.Module,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one RK4 step (non-symplectic)."""
        dt = self.dt
        
        def vector_field(q_val: torch.Tensor, p_val: torch.Tensor):
            """Compute (dq/dt, dp/dt) = (∂H/∂p, -∂H/∂q)."""
            q_v = q_val.clone().requires_grad_(True)
            p_v = p_val.clone().requires_grad_(True)
            
            if edge_index is not None and hasattr(hamiltonian, 'compute_gradients'):
                grad_q, grad_p = hamiltonian.compute_gradients(q_v, p_v, edge_index)
            elif hasattr(hamiltonian, 'compute_gradients'):
                grad_q, grad_p = hamiltonian.compute_gradients(q_v, p_v)
            else:
                H = hamiltonian(q_v, p_v, sum_over_particles=True)
                grad_q, grad_p = torch.autograd.grad(
                    H, [q_v, p_v],
                    create_graph=False,
                )
            
            # Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
            dq_dt = grad_p.detach()
            dp_dt = -grad_q.detach()
            
            return dq_dt, dp_dt
        
        # k1 at current state
        dq1, dp1 = vector_field(q, p)
        
        # k2 at midpoint using k1
        q2 = q + 0.5 * dt * dq1
        p2 = p + 0.5 * dt * dp1
        dq2, dp2 = vector_field(q2, p2)
        
        # k3 at midpoint using k2
        q3 = q + 0.5 * dt * dq2
        p3 = p + 0.5 * dt * dp2
        dq3, dp3 = vector_field(q3, p3)
        
        # k4 at end using k3
        q4 = q + dt * dq3
        p4 = p + dt * dp3
        dq4, dp4 = vector_field(q4, p4)
        
        # Weighted average
        q_new = q + (dt / 6.0) * (dq1 + 2*dq2 + 2*dq3 + dq4)
        p_new = p + (dt / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
        
        return q_new, p_new


class AdaptiveRK45(BaseIntegrator):
    """
    Adaptive Runge-Kutta-Fehlberg (RK45) integrator.
    
    Uses embedded RK4/RK5 pair for error estimation and
    adaptive step size control.
    
    NOT symplectic, but useful for comparison with adaptive methods.
    """
    
    def __init__(
        self,
        dt: float = 1.0,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_dt: float = 100.0,
        min_dt: float = 1e-6,
    ):
        super().__init__(dt=dt)
        self.rtol = rtol
        self.atol = atol
        self.max_dt = max_dt
        self.min_dt = min_dt
        
        # RK45 Butcher tableau (Dormand-Prince)
        self.c = torch.tensor([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=torch.float64)
        self.a = [
            [],
            [1/5],
            [3/40, 9/40],
            [44/45, -56/15, 32/9],
            [19372/6561, -25360/2187, 64448/6561, -212/729],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
        ]
        # 5th order weights
        self.b5 = torch.tensor([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=torch.float64)
        # 4th order weights
        self.b4 = torch.tensor([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=torch.float64)
    
    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        hamiltonian: nn.Module,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one adaptive RK45 step.
        
        Note: This implementation uses fixed dt for simplicity.
        Full adaptive stepping would adjust dt based on error.
        """
        # For simplicity, use fixed-step RK4 here
        # Full adaptive implementation would iterate until error < tol
        rk4 = RK4(dt=float(self.dt))
        return rk4.step(q, p, hamiltonian, edge_index)


def compare_integrators(
    q0: torch.Tensor,
    p0: torch.Tensor,
    hamiltonian: nn.Module,
    n_steps: int,
    dt: float,
    integrator_names: list = ["symplectic_euler", "vanilla_euler", "rk4"],
) -> dict:
    """
    Compare different integrators on the same initial conditions.
    
    Returns trajectories and energy time series for each integrator.
    
    Args:
        q0: Initial positions
        p0: Initial momenta
        hamiltonian: Hamiltonian module
        n_steps: Number of steps
        dt: Time step
        integrator_names: List of integrator names to compare
        
    Returns:
        Dictionary with results for each integrator
    """
    from .symplectic import SymplecticEuler, StormerVerlet
    
    integrators = {
        "symplectic_euler": SymplecticEuler(dt=dt),
        "stormer_verlet": StormerVerlet(dt=dt),
        "vanilla_euler": VanillaEuler(dt=dt),
        "rk4": RK4(dt=dt),
    }
    
    results = {}
    
    for name in integrator_names:
        if name not in integrators:
            continue
            
        integrator = integrators[name]
        q_traj, p_traj = integrator.integrate(
            q0.clone(), p0.clone(), hamiltonian, n_steps, store_trajectory=True
        )
        
        # Compute energy at each step
        energies = []
        for i in range(len(q_traj)):
            with torch.no_grad():
                E = hamiltonian(q_traj[i], p_traj[i], sum_over_particles=True)
                energies.append(E.item())
        
        results[name] = {
            "q_trajectory": q_traj,
            "p_trajectory": p_traj,
            "energy": torch.tensor(energies),
            "energy_drift": torch.tensor(energies) - energies[0],
        }
    
    return results
