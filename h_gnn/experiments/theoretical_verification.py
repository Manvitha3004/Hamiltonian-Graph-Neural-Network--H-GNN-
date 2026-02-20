"""
Theoretical analysis supporting the H-GNN theorems.

This module provides computational verification of the theoretical bounds
derived in THEORETICAL_FOUNDATIONS.md.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from pathlib import Path


def compute_snr_threshold(
    epsilon_values: List[float],
    T: int = 50,
    dt: float = 60.0,
    integration_noise: float = 1e-4,
    sampling_noise: float = 1e-3,
    grad_phi_norm: float = 0.01,
) -> Dict[str, List[float]]:
    """
    Compute theoretical SNR for different epsilon values.
    
    SNR = (epsilon * T * ||grad_Phi||) / sigma_noise
    
    where sigma_noise = sqrt(T) * sigma_integration + sigma_sampling
    
    Args:
        epsilon_values: List of interaction strengths
        T: Prediction horizon
        dt: Time step
        integration_noise: Per-step integration error std
        sampling_noise: Sampling noise std
        grad_phi_norm: Typical gradient norm of Phi
        
    Returns:
        Dictionary with epsilon, snr, and predicted success
    """
    sigma_noise = np.sqrt(T) * integration_noise + sampling_noise
    
    results = {
        "epsilon": epsilon_values,
        "snr": [],
        "predicted_success": [],
    }
    
    for eps in epsilon_values:
        snr = (eps * T * grad_phi_norm) / sigma_noise
        results["snr"].append(snr)
        results["predicted_success"].append(snr > 1.0)
    
    return results


def verify_symplectic_bound(
    n_steps: int = 500,
    dt: float = 60.0,
    mu: float = 398600.4418,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Verify that symplectic integration bounds energy drift.
    
    Compares:
    - Störmer-Verlet (symplectic, order 2): E(t) bounded
    - RK4 (non-symplectic, order 4): E(t) drifts linearly
    
    Returns:
        steps, symplectic_drift, nonsymplectic_drift
    """
    # Simple harmonic oscillator as test case
    omega = 0.001  # Angular frequency
    
    # Initial conditions
    q0, p0 = 1.0, 0.0
    
    # Hamiltonian: H = p^2/2 + omega^2 * q^2/2
    def H(q, p):
        return 0.5 * p**2 + 0.5 * omega**2 * q**2
    
    E0 = H(q0, p0)
    
    # Störmer-Verlet
    q_sv, p_sv = q0, p0
    sv_drift = [0.0]
    
    for _ in range(n_steps):
        # Half step p
        p_sv = p_sv - 0.5 * dt * omega**2 * q_sv
        # Full step q
        q_sv = q_sv + dt * p_sv
        # Half step p
        p_sv = p_sv - 0.5 * dt * omega**2 * q_sv
        
        sv_drift.append(abs(H(q_sv, p_sv) - E0) / E0)
    
    # RK4 (non-symplectic)
    q_rk, p_rk = q0, p0
    rk_drift = [0.0]
    
    def derivs(q, p):
        return p, -omega**2 * q
    
    for _ in range(n_steps):
        k1q, k1p = derivs(q_rk, p_rk)
        k2q, k2p = derivs(q_rk + 0.5*dt*k1q, p_rk + 0.5*dt*k1p)
        k3q, k3p = derivs(q_rk + 0.5*dt*k2q, p_rk + 0.5*dt*k2p)
        k4q, k4p = derivs(q_rk + dt*k3q, p_rk + dt*k3p)
        
        q_rk = q_rk + (dt/6) * (k1q + 2*k2q + 2*k3q + k4q)
        p_rk = p_rk + (dt/6) * (k1p + 2*k2p + 2*k3p + k4p)
        
        rk_drift.append(abs(H(q_rk, p_rk) - E0) / E0)
    
    steps = np.arange(n_steps + 1)
    return steps, np.array(sv_drift), np.array(rk_drift)


def verify_dt_squared_scaling(
    dt_values: List[float] = [10.0, 30.0, 60.0, 120.0],
    n_steps: int = 100,
) -> Dict[str, List[float]]:
    """
    Verify that Störmer-Verlet error scales as dt^2.
    
    Returns:
        Dictionary with dt values and max energy drift
    """
    omega = 0.001
    
    results = {"dt": dt_values, "max_drift": [], "theoretical_dt2": []}
    
    for dt in dt_values:
        q, p = 1.0, 0.0
        E0 = 0.5 * p**2 + 0.5 * omega**2 * q**2
        max_drift = 0.0
        
        for _ in range(n_steps):
            p = p - 0.5 * dt * omega**2 * q
            q = q + dt * p
            p = p - 0.5 * dt * omega**2 * q
            
            E = 0.5 * p**2 + 0.5 * omega**2 * q**2
            drift = abs(E - E0) / E0
            max_drift = max(max_drift, drift)
        
        results["max_drift"].append(max_drift)
        results["theoretical_dt2"].append(dt**2 * 1e-8)  # Scaled constant
    
    return results


def plot_theoretical_verification():
    """Generate plots verifying theoretical predictions."""
    
    fig_dir = Path(__file__).parent.parent.parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # 1. SNR vs Epsilon
    print("Computing SNR threshold verification...")
    epsilons = [1, 2, 5, 10, 25, 50, 100, 200]
    snr_results = compute_snr_threshold(epsilons)
    
    # Observed improvements from experiments
    observed = {
        1: -1.9, 5: 15.0, 10: 18.8, 25: 23.4, 50: 20.9, 100: 13.4, 200: 15.9
    }
    
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    
    ax1.semilogy(epsilons, snr_results["snr"], 'o-', color='#3498db', 
                 linewidth=2, markersize=8, label='Theoretical SNR')
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='SNR = 1 (threshold)')
    ax1.fill_between([0.5, 5], [0.1, 0.1], [1, 1], alpha=0.2, color='red')
    ax1.set_xlabel('Interaction Strength (epsilon)')
    ax1.set_ylabel('Signal-to-Noise Ratio', color='#3498db')
    ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_xlim(0.8, 250)
    ax1.set_ylim(0.1, 100)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    obs_eps = list(observed.keys())
    obs_imp = list(observed.values())
    ax2.plot(obs_eps, obs_imp, 's--', color='#2ecc71', markersize=7, 
             linewidth=1.5, alpha=0.8, label='Observed Improvement')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Observed Improvement (%)', color='#2ecc71')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    ax2.legend(loc='upper right')
    
    ax1.set_title('Theorem 1: Learnability Threshold')
    ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "theorem1_snr_threshold.png", dpi=300)
    plt.savefig(fig_dir / "theorem1_snr_threshold.pdf")
    print("Saved: theorem1_snr_threshold.png/pdf")
    plt.close()
    
    # 2. Symplectic vs Non-symplectic drift
    print("Computing symplectic bound verification...")
    steps, sv_drift, rk_drift = verify_symplectic_bound(n_steps=500, dt=10.0)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.semilogy(steps, sv_drift + 1e-16, '-', color='#2ecc71', linewidth=2, 
                label='Störmer-Verlet (Symplectic)')
    ax.semilogy(steps, rk_drift + 1e-16, '-', color='#e74c3c', linewidth=2,
                label='RK4 (Non-symplectic)')
    
    # Add theoretical linear drift line
    linear_drift = 1e-10 * steps
    ax.semilogy(steps[1:], linear_drift[1:], '--', color='#95a5a6', linewidth=1.5,
                label='Linear drift (theory)')
    
    ax.set_xlabel('Integration Steps')
    ax.set_ylabel('Relative Energy Drift')
    ax.set_title('Theorem 2: Symplectic Bounds Energy Drift')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(1e-16, 1e-5)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "theorem2_symplectic_bound.png", dpi=300)
    plt.savefig(fig_dir / "theorem2_symplectic_bound.pdf")
    print("Saved: theorem2_symplectic_bound.png/pdf")
    plt.close()
    
    # 3. dt^2 scaling verification
    print("Computing dt^2 scaling verification...")
    dt_results = verify_dt_squared_scaling()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.loglog(dt_results["dt"], dt_results["max_drift"], 'o-', color='#3498db',
              linewidth=2, markersize=10, label='Observed')
    
    # Theoretical dt^2 line
    dts = np.array(dt_results["dt"])
    theoretical = dt_results["max_drift"][0] * (dts / dts[0])**2
    ax.loglog(dts, theoretical, '--', color='#e74c3c', linewidth=2,
              label='dt² scaling (theory)')
    
    ax.set_xlabel('Time Step (dt)')
    ax.set_ylabel('Max Energy Drift')
    ax.set_title('Theorem 3: Störmer-Verlet Error ~ O(dt²)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "theorem3_dt_squared.png", dpi=300)
    plt.savefig(fig_dir / "theorem3_dt_squared.pdf")
    print("Saved: theorem3_dt_squared.png/pdf")
    plt.close()
    
    print("\nAll theoretical verification plots generated.")


if __name__ == "__main__":
    plot_theoretical_verification()
