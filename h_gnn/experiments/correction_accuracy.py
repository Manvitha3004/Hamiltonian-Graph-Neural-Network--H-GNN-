"""
Correction Accuracy Quantification.

Measures: ||∇Φ_θ - ∇Φ_true|| / ||∇Φ_true||

This proves the model actually learns the correction, 
not just compensates indirectly.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List

from h_gnn.models import HGNN, TwoBodyHamiltonian
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet


def quantify_correction_accuracy(
    epsilon: float = 50.0,
    n_particles: int = 5,
    n_samples: int = 100,
    seed: int = 42,
) -> Dict:
    """
    Quantify how accurately the learned Φ matches the true interaction.
    
    Computes: ||∇Φ_θ - ∇Φ_true|| / ||∇Φ_true||
    
    Where:
    - ∇Φ_θ = ∇H_θ - ∇H_twobody (learned correction)
    - ∇Φ_true = ∇H_multibody - ∇H_twobody (true correction)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("CORRECTION ACCURACY QUANTIFICATION")
    print("=" * 70)
    print(f"ε={epsilon}, N={n_particles}, n_samples={n_samples}")
    print()
    
    # Create Hamiltonians
    gt_hamiltonian = MultiBodyHamiltonian(epsilon=epsilon)
    tb_hamiltonian = TwoBodyHamiltonian()
    
    # Create H-GNN
    hgnn = HGNN(
        mu=398600.4418,
        interaction_strength=1.0,
        interaction_radius=5000.0,
        hidden_dims=[128, 128, 64],
        integrator_type='stormer_verlet',
        dt=60.0,
        softplus_output=False,
        normalize_features=True,
    ).double()
    
    # Load pretrained model if available
    model_path = Path("hgnn_multibody.pt")
    if model_path.exists():
        hgnn.load_state_dict(torch.load(model_path, weights_only=True))
        print("Loaded pretrained H-GNN model")
    else:
        print("WARNING: No pretrained model found. Using random weights.")
    
    hgnn.eval()
    
    # Build edge index
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Collect statistics
    relative_errors = []
    absolute_errors = []
    true_magnitudes = []
    learned_magnitudes = []
    cosine_similarities = []
    
    print("Sampling correction gradients...")
    
    for sample_idx in range(n_samples):
        # Random state
        np.random.seed(seed + sample_idx)
        r = 7000 + 500 * np.random.randn(n_particles)
        theta = np.linspace(0, 2*np.pi, n_particles, endpoint=False) + 0.1 * np.random.randn(n_particles)
        
        q = torch.tensor([
            [r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i]), 50.0 * np.random.randn()]
            for i in range(n_particles)
        ], dtype=torch.float64)
        
        v_circ = np.sqrt(398600.4418 / r)
        p = torch.tensor([
            [v_circ[i] * (-np.sin(theta[i])), v_circ[i] * np.cos(theta[i]), 0.1 * np.random.randn()]
            for i in range(n_particles)
        ], dtype=torch.float64)
        
        with torch.no_grad():
            # True correction: ∇H_multibody - ∇H_twobody
            grad_q_mb, _ = gt_hamiltonian.compute_gradients(q, p, edge_index)
            grad_q_tb, _ = tb_hamiltonian.compute_gradients(q, p)
            grad_phi_true = grad_q_mb - grad_q_tb
            
            # Learned correction: ∇H_θ - ∇H_twobody
            # Note: H_θ = H_twobody + λΦ, so ∇H_θ - ∇H_twobody = λ∇Φ
            grad_q_hgnn, _ = hgnn.hamiltonian.compute_gradients(q, p, edge_index)
            grad_phi_learned = grad_q_hgnn - grad_q_tb
        
        # Compute metrics
        true_mag = torch.norm(grad_phi_true).item()
        learned_mag = torch.norm(grad_phi_learned).item()
        
        if true_mag > 1e-12:
            abs_error = torch.norm(grad_phi_learned - grad_phi_true).item()
            rel_error = abs_error / true_mag
            
            # Cosine similarity
            cos_sim = torch.sum(grad_phi_learned * grad_phi_true).item() / (
                learned_mag * true_mag + 1e-12
            )
            
            relative_errors.append(rel_error)
            absolute_errors.append(abs_error)
            true_magnitudes.append(true_mag)
            learned_magnitudes.append(learned_mag)
            cosine_similarities.append(cos_sim)
        
        if sample_idx % 20 == 0 and sample_idx > 0:
            print(f"  Sample {sample_idx}: rel_err={rel_error:.3f}, cos_sim={cos_sim:.3f}")
    
    # Aggregate statistics
    results = {
        'relative_error': {
            'mean': np.mean(relative_errors),
            'std': np.std(relative_errors),
            'median': np.median(relative_errors),
            'min': np.min(relative_errors),
            'max': np.max(relative_errors),
        },
        'cosine_similarity': {
            'mean': np.mean(cosine_similarities),
            'std': np.std(cosine_similarities),
            'median': np.median(cosine_similarities),
        },
        'magnitude_ratio': {
            'mean': np.mean([l/t for l, t in zip(learned_magnitudes, true_magnitudes)]),
            'std': np.std([l/t for l, t in zip(learned_magnitudes, true_magnitudes)]),
        },
        'true_magnitude': {
            'mean': np.mean(true_magnitudes),
            'std': np.std(true_magnitudes),
        },
        'learned_magnitude': {
            'mean': np.mean(learned_magnitudes),
            'std': np.std(learned_magnitudes),
        },
        'raw_data': {
            'relative_errors': relative_errors,
            'cosine_similarities': cosine_similarities,
        }
    }
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n1. RELATIVE ERROR: ||∇Φ_θ - ∇Φ_true|| / ||∇Φ_true||")
    print("-" * 50)
    print(f"   Mean:   {results['relative_error']['mean']:.3f}")
    print(f"   Std:    {results['relative_error']['std']:.3f}")
    print(f"   Median: {results['relative_error']['median']:.3f}")
    print(f"   Range:  [{results['relative_error']['min']:.3f}, {results['relative_error']['max']:.3f}]")
    
    print("\n2. COSINE SIMILARITY: ∇Φ_θ · ∇Φ_true / (||∇Φ_θ|| ||∇Φ_true||)")
    print("-" * 50)
    print(f"   Mean:   {results['cosine_similarity']['mean']:.3f}")
    print(f"   Std:    {results['cosine_similarity']['std']:.3f}")
    print(f"   Median: {results['cosine_similarity']['median']:.3f}")
    
    print("\n3. MAGNITUDE RATIO: ||∇Φ_θ|| / ||∇Φ_true||")
    print("-" * 50)
    print(f"   Mean:   {results['magnitude_ratio']['mean']:.3f}")
    print(f"   Std:    {results['magnitude_ratio']['std']:.3f}")
    
    print()
    print("INTERPRETATION:")
    print("-" * 50)
    
    rel_err = results['relative_error']['mean']
    cos_sim = results['cosine_similarity']['mean']
    
    if rel_err < 0.5 and cos_sim > 0.5:
        print("✓ Model LEARNS the true correction (not just indirect compensation)")
        print(f"  - {100*(1-rel_err):.0f}% of correction magnitude is accurate")
        print(f"  - {100*cos_sim:.0f}% directional alignment")
    elif rel_err < 1.0 and cos_sim > 0:
        print("~ Model partially learns the correction")
        print(f"  - Some signal captured but significant error remains")
    else:
        print("✗ Model does NOT learn the true correction")
        print(f"  - Improvement may come from indirect compensation")
    
    # Save results
    output_dir = Path("results/correction_accuracy")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Don't save raw data in JSON (too large)
    results_to_save = {k: v for k, v in results.items() if k != 'raw_data'}
    
    with open(output_dir / "correction_accuracy_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/correction_accuracy_results.json")
    
    return results


if __name__ == "__main__":
    results = quantify_correction_accuracy(
        epsilon=50.0,
        n_particles=5,
        n_samples=100,
    )
