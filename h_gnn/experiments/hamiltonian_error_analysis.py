"""
Hamiltonian Error Analysis.

This diagnostic measures how well the learned Hamiltonian matches the true
Hamiltonian across different energy levels. If error grows away from the
training region, it confirms global underfitting.

Key measurements:
1. |H_model - H_true| at various energy levels
2. Error distribution across phase space
3. Correlation between training state distribution and accuracy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from h_gnn.models import HGNN
from h_gnn.models.multibody import MultiBodyHamiltonian


def generate_random_states(
    n_particles: int,
    n_samples: int,
    energy_range: Tuple[float, float] = (0.5, 2.0),
    seed: int = 42,
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Generate random states across different energy levels.
    
    Uses orbital mechanics to create states at controlled energy levels.
    Energy scaling factor controls how far from circular orbit.
    
    Args:
        n_particles: Number of particles
        n_samples: Number of state samples
        energy_range: Range of energy scale factors (1.0 = circular orbit)
        seed: Random seed
        
    Returns:
        List of (q, p, energy_scale) tuples
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    states = []
    energy_scales = np.linspace(energy_range[0], energy_range[1], n_samples)
    
    for scale in energy_scales:
        # Base circular orbit configuration
        r_base = 7000.0  # km, typical LEO
        r = r_base * (0.9 + 0.2 * np.random.rand(n_particles))
        theta = np.linspace(0, 2*np.pi, n_particles, endpoint=False)
        theta += 0.2 * np.random.randn(n_particles)
        
        # Positions
        q = np.zeros((n_particles, 3))
        q[:, 0] = r * np.cos(theta)
        q[:, 1] = r * np.sin(theta)
        q[:, 2] = 100.0 * np.random.randn(n_particles)
        
        # Circular velocity
        mu = 398600.4418
        v_circ = np.sqrt(mu / r)
        
        # Scale velocity by energy factor
        # scale > 1: higher energy (more elliptical, moving away)
        # scale < 1: lower energy (tighter orbit)
        v = v_circ * scale
        
        # Momenta (tangent to orbit with scaling)
        p = np.zeros((n_particles, 3))
        p[:, 0] = -v * np.sin(theta)
        p[:, 1] = v * np.cos(theta)
        p[:, 2] = 0.1 * np.random.randn(n_particles)
        
        q_tensor = torch.tensor(q, dtype=torch.float32)
        p_tensor = torch.tensor(p, dtype=torch.float32)
        
        states.append((q_tensor, p_tensor, scale))
    
    return states


def compute_true_hamiltonian(
    q: torch.Tensor,
    p: torch.Tensor,
    epsilon: float = 50.0,
    mu: float = 398600.4418,
) -> float:
    """
    Compute true Hamiltonian: H = sum(H_kepler) + sum(phi_true).
    
    Uses MultiBodyHamiltonian for consistent ground truth.
    """
    gt_hamiltonian = MultiBodyHamiltonian(epsilon=epsilon, mu=mu)
    
    with torch.no_grad():
        q_d = q.double()
        p_d = p.double()
        # Use the forward method which computes full Hamiltonian
        H = gt_hamiltonian(q_d, p_d, sum_over_particles=True)
    
    return H.item()


def compute_model_hamiltonian(
    model: HGNN,
    q: torch.Tensor,
    p: torch.Tensor,
) -> float:
    """
    Compute model's Hamiltonian value.
    """
    with torch.no_grad():
        H = model.compute_hamiltonian(q, p)
    
    return H.item()


def analyze_hamiltonian_error(
    model: HGNN,
    n_particles: int = 5,
    n_samples: int = 20,
    energy_range: Tuple[float, float] = (0.3, 2.5),
    epsilon: float = 50.0,
) -> Dict:
    """
    Analyze Hamiltonian error across energy levels.
    
    Returns dictionary with error statistics.
    """
    print("=" * 60)
    print("HAMILTONIAN ERROR ANALYSIS")
    print("=" * 60)
    print(f"Testing {n_samples} states across energy scales {energy_range}")
    print()
    
    # Generate states
    states = generate_random_states(n_particles, n_samples, energy_range)
    
    results = {
        "energy_scales": [],
        "true_H": [],
        "model_H": [],
        "abs_error": [],
        "rel_error": [],
    }
    
    print(f"{'Scale':>8} {'H_true':>14} {'H_model':>14} {'|Error|':>12} {'Rel %':>8}")
    print("-" * 60)
    
    for q, p, scale in states:
        H_true = compute_true_hamiltonian(q, p, epsilon=epsilon)
        H_model = compute_model_hamiltonian(model, q, p)
        
        abs_err = abs(H_model - H_true)
        rel_err = abs_err / (abs(H_true) + 1e-10) * 100
        
        results["energy_scales"].append(scale)
        results["true_H"].append(H_true)
        results["model_H"].append(H_model)
        results["abs_error"].append(abs_err)
        results["rel_error"].append(rel_err)
        
        print(f"{scale:8.2f} {H_true:14.2e} {H_model:14.2e} {abs_err:12.2e} {rel_err:7.2f}%")
    
    print()
    print("SUMMARY")
    print("-" * 40)
    print(f"Mean absolute error: {np.mean(results['abs_error']):.2e}")
    print(f"Max absolute error:  {np.max(results['abs_error']):.2e}")
    print(f"Mean relative error: {np.mean(results['rel_error']):.2f}%")
    print(f"Max relative error:  {np.max(results['rel_error']):.2f}%")
    
    # Check monotonicity: does error grow away from training region?
    scales = np.array(results["energy_scales"])
    errors = np.array(results["rel_error"])
    
    # Training typically near scale=1.0
    training_region = (scales >= 0.8) & (scales <= 1.2)
    outside_region = ~training_region
    
    if np.sum(training_region) > 0 and np.sum(outside_region) > 0:
        training_err = np.mean(errors[training_region])
        outside_err = np.mean(errors[outside_region])
        
        print()
        print("REGIONAL ANALYSIS (training region: scale 0.8-1.2)")
        print(f"  Mean error in training region:  {training_err:.2f}%")
        print(f"  Mean error outside region:      {outside_err:.2f}%")
        
        if outside_err > training_err * 1.5:
            print("  -> CONFIRMED: Error grows outside training distribution")
            print("     This indicates global underfitting")
        else:
            print("  -> Error is similar across energy levels")
            print("     Model generalizes reasonably")
    
    return results


def plot_hamiltonian_error(results: Dict, save_path: Optional[str] = None):
    """
    Plot Hamiltonian error vs energy scale.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    scales = np.array(results["energy_scales"])
    true_H = np.array(results["true_H"])
    model_H = np.array(results["model_H"])
    abs_err = np.array(results["abs_error"])
    rel_err = np.array(results["rel_error"])
    
    # 1. H_true vs H_model scatter
    ax = axes[0]
    ax.scatter(true_H, model_H, c=scales, cmap='coolwarm', alpha=0.7)
    
    # Perfect match line
    H_range = [min(true_H.min(), model_H.min()), max(true_H.max(), model_H.max())]
    ax.plot(H_range, H_range, 'k--', alpha=0.5, label='Perfect match')
    
    ax.set_xlabel('H_true')
    ax.set_ylabel('H_model')
    ax.set_title('Model vs True Hamiltonian')
    ax.legend()
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(scales.min(), scales.max()))
    plt.colorbar(sm, ax=ax, label='Energy scale')
    
    # 2. Absolute error vs energy scale
    ax = axes[1]
    ax.semilogy(scales, abs_err, 'o-', color='tab:blue')
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='Nominal training')
    ax.axvspan(0.8, 1.2, alpha=0.2, color='green', label='Training region')
    ax.set_xlabel('Energy Scale')
    ax.set_ylabel('|H_model - H_true|')
    ax.set_title('Absolute Error vs Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Relative error vs energy scale
    ax = axes[2]
    ax.plot(scales, rel_err, 'o-', color='tab:orange')
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvspan(0.8, 1.2, alpha=0.2, color='green', label='Training region')
    ax.set_xlabel('Energy Scale')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Relative Error vs Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def run_analysis(
    model_path: Optional[str] = None,
    n_particles: int = 5,
    n_samples: int = 25,
    epsilon: float = 50.0,
    save_plot: bool = True,
):
    """
    Run complete Hamiltonian error analysis.
    
    Args:
        model_path: Path to trained model, or None to use fresh model
        n_particles: Number of particles
        n_samples: Number of energy level samples
        epsilon: Ground truth interaction strength
        save_plot: Whether to save plot
    """
    # Create model
    model = HGNN(
        mu=398600.4418,
        interaction_strength=1.0,  # phi_scale with num_edges normalization
        interaction_radius=5000.0,
        hidden_dims=[64, 64, 32],
        integrator_type='stormer_verlet',
        dt=60.0,
        softplus_output=False,
        normalize_features=True,
        create_graph=False,
        init_gain=1.0,
    )
    
    # Load trained weights if provided
    if model_path and Path(model_path).exists():
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    else:
        print("Using untrained model (for comparison baseline)")
    
    model.eval()
    
    # Run analysis
    results = analyze_hamiltonian_error(
        model,
        n_particles=n_particles,
        n_samples=n_samples,
        energy_range=(0.3, 2.5),
        epsilon=epsilon,
    )
    
    # Plot
    if save_plot:
        plot_path = Path(__file__).parent / "hamiltonian_error_plot.png"
        plot_hamiltonian_error(results, save_path=str(plot_path))
    else:
        plot_hamiltonian_error(results)
    
    return results


if __name__ == "__main__":
    # Try to load trained model
    project_root = Path(__file__).parent.parent.parent
    model_paths = [
        project_root / "hgnn_multiscale.pt",
        project_root / "hgnn_best.pt",
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = str(path)
            break
    
    results = run_analysis(
        model_path=model_path,
        n_particles=5,
        n_samples=25,
        epsilon=50.0,
        save_plot=True,
    )
