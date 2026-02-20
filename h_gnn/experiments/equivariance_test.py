"""
SO(3) Equivariance Test.

Validates that the Hamiltonian is SO(3) invariant and
the dynamics are SO(3) equivariant.

Key tests:
1. H(Rz) ≈ H(z) for all R ∈ SO(3)
2. Ψ(Rz) ≈ RΨ(z) where Ψ is the flow map
"""

import torch
import numpy as np
from typing import Dict, Tuple
from scipy.spatial.transform import Rotation


def random_rotation_matrix(device: str = "cpu") -> torch.Tensor:
    """Generate random SO(3) rotation matrix."""
    R = Rotation.random().as_matrix()
    return torch.tensor(R, dtype=torch.float64, device=device)


def apply_rotation(
    q: torch.Tensor,
    p: torch.Tensor,
    R: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotation R to phase space state (q, p).
    
    Both position and momentum transform as vectors.
    
    Args:
        q: Positions (N, 3)
        p: Momenta (N, 3)
        R: Rotation matrix (3, 3)
        
    Returns:
        Rotated (q_rot, p_rot)
    """
    q_rot = torch.matmul(q, R.T)
    p_rot = torch.matmul(p, R.T)
    return q_rot, p_rot


def test_hamiltonian_invariance(
    hamiltonian: torch.nn.Module,
    q: torch.Tensor,
    p: torch.Tensor,
    n_rotations: int = 10,
    tolerance: float = 1e-10,
) -> Dict[str, float]:
    """
    Test SO(3) invariance of Hamiltonian.
    
    Verifies: H(Rq, Rp) = H(q, p) for random R ∈ SO(3)
    
    Args:
        hamiltonian: Hamiltonian module
        q: Test positions
        p: Test momenta
        n_rotations: Number of random rotations to test
        tolerance: Numerical tolerance
        
    Returns:
        Dict with test results
    """
    device = q.device
    
    # Compute original Hamiltonian
    with torch.no_grad():
        H_original = hamiltonian(q, p, sum_over_particles=True)
    
    errors = []
    
    for _ in range(n_rotations):
        R = random_rotation_matrix(device)
        q_rot, p_rot = apply_rotation(q, p, R)
        
        with torch.no_grad():
            H_rotated = hamiltonian(q_rot, p_rot, sum_over_particles=True)
        
        rel_error = torch.abs(H_rotated - H_original) / (torch.abs(H_original) + 1e-10)
        errors.append(rel_error.item())
    
    max_error = max(errors)
    mean_error = np.mean(errors)
    
    passed = max_error < tolerance
    
    return {
        "passed": passed,
        "max_relative_error": max_error,
        "mean_relative_error": mean_error,
        "tolerance": tolerance,
    }


def test_dynamics_equivariance(
    model: torch.nn.Module,
    q: torch.Tensor,
    p: torch.Tensor,
    n_steps: int = 10,
    n_rotations: int = 5,
    tolerance: float = 1e-6,
) -> Dict[str, float]:
    """
    Test SO(3) equivariance of dynamics.
    
    Verifies: Ψ_t(Rz) ≈ R Ψ_t(z)
    
    Where Ψ_t is the t-step flow map.
    
    Args:
        model: H-GNN model
        q: Test positions
        p: Test momenta
        n_steps: Integration steps
        n_rotations: Number of rotations to test
        tolerance: Numerical tolerance
        
    Returns:
        Dict with test results
    """
    device = q.device
    model.eval()
    
    errors_q = []
    errors_p = []
    
    for _ in range(n_rotations):
        R = random_rotation_matrix(device)
        
        # Method 1: Integrate then rotate
        with torch.no_grad():
            q_final, p_final = model(q, p, n_steps=n_steps)
        
        q_final_rot, p_final_rot = apply_rotation(q_final, p_final, R)
        
        # Method 2: Rotate then integrate
        q_rot, p_rot = apply_rotation(q, p, R)
        
        with torch.no_grad():
            q_rot_final, p_rot_final = model(q_rot, p_rot, n_steps=n_steps)
        
        # Compare
        error_q = torch.mean(torch.abs(q_rot_final - q_final_rot)).item()
        error_p = torch.mean(torch.abs(p_rot_final - p_final_rot)).item()
        
        errors_q.append(error_q)
        errors_p.append(error_p)
    
    max_error = max(max(errors_q), max(errors_p))
    passed = max_error < tolerance
    
    return {
        "passed": passed,
        "max_position_error": max(errors_q),
        "max_momentum_error": max(errors_p),
        "mean_position_error": np.mean(errors_q),
        "mean_momentum_error": np.mean(errors_p),
        "tolerance": tolerance,
    }


def test_interaction_symmetry(
    interaction: torch.nn.Module,
    n_pairs: int = 100,
    tolerance: float = 1e-10,
) -> Dict[str, float]:
    """
    Test symmetry of interaction potential.
    
    Verifies: Φ(zᵢ, zⱼ) = Φ(zⱼ, zᵢ)
    
    Args:
        interaction: Interaction potential module
        n_pairs: Number of random pairs to test
        tolerance: Numerical tolerance
        
    Returns:
        Dict with test results
    """
    errors = []
    
    for _ in range(n_pairs):
        # Random states
        q_i = torch.randn(1, 3, dtype=torch.float64) * 7000
        p_i = torch.randn(1, 3, dtype=torch.float64) * 10
        q_j = torch.randn(1, 3, dtype=torch.float64) * 7000
        p_j = torch.randn(1, 3, dtype=torch.float64) * 10
        
        with torch.no_grad():
            phi_ij = interaction(q_i, p_i, q_j, p_j)
            phi_ji = interaction(q_j, p_j, q_i, p_i)
        
        error = torch.abs(phi_ij - phi_ji).item()
        errors.append(error)
    
    max_error = max(errors)
    passed = max_error < tolerance
    
    return {
        "passed": passed,
        "max_asymmetry": max_error,
        "mean_asymmetry": np.mean(errors),
        "tolerance": tolerance,
    }


def run_equivariance_tests(
    model: torch.nn.Module,
    config: Dict,
) -> Dict[str, Dict]:
    """
    Run all equivariance tests.
    
    Args:
        model: H-GNN model
        config: Test configuration
        
    Returns:
        Dict of all test results
    """
    print("\n" + "=" * 60)
    print("SO(3) Equivariance Tests")
    print("=" * 60)
    
    # Generate test data
    from h_gnn.data import SyntheticCatalog
    
    catalog = SyntheticCatalog(seed=42)
    q, p = catalog.generate_training_set(n_objects=20)
    
    results = {}
    
    # Test 1: Hamiltonian invariance
    print("\n1. Hamiltonian SO(3) Invariance")
    print("-" * 40)
    
    result = test_hamiltonian_invariance(
        model.H_orbit, q, p,
        n_rotations=10,
        tolerance=1e-10,
    )
    results["hamiltonian_invariance"] = result
    
    status = "✓ PASSED" if result["passed"] else "✗ FAILED"
    print(f"   {status}")
    print(f"   Max relative error: {result['max_relative_error']:.2e}")
    
    # Test 2: Dynamics equivariance
    print("\n2. Dynamics SO(3) Equivariance")
    print("-" * 40)
    
    result = test_dynamics_equivariance(
        model, q, p,
        n_steps=10,
        n_rotations=5,
        tolerance=1e-4,
    )
    results["dynamics_equivariance"] = result
    
    status = "✓ PASSED" if result["passed"] else "✗ FAILED"
    print(f"   {status}")
    print(f"   Max position error: {result['max_position_error']:.2e}")
    print(f"   Max momentum error: {result['max_momentum_error']:.2e}")
    
    # Test 3: Interaction symmetry
    print("\n3. Interaction Potential Symmetry")
    print("-" * 40)
    
    result = test_interaction_symmetry(
        model.interaction,
        n_pairs=100,
        tolerance=1e-8,
    )
    results["interaction_symmetry"] = result
    
    status = "✓ PASSED" if result["passed"] else "✗ FAILED"
    print(f"   {status}")
    print(f"   Max asymmetry: {result['max_asymmetry']:.2e}")
    
    # Summary
    print("\n" + "=" * 60)
    all_passed = all(r.get("passed", False) for r in results.values())
    
    if all_passed:
        print("All equivariance tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
        for name, result in results.items():
            if not result.get("passed", False):
                print(f"  - {name}")
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    from h_gnn.models import HGNN
    
    # Create model
    model = HGNN(
        mu=398600.4418,
        interaction_strength=0.01,
        interaction_radius=1000.0,
        hidden_dims=[64, 64, 32],
        dt=60.0,
    ).double()  # Convert to float64 for orbital mechanics precision
    
    config = {}
    results = run_equivariance_tests(model, config)
