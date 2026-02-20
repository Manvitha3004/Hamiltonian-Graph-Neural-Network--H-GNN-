"""
Rare Regime Evaluation Experiment.

Tests whether Hamiltonian structure improves generalization
from training distribution (equatorial orbits) to rare
test distribution (polar orbits).

Key metric: Does H-GNN degrade less than baselines when
tested on out-of-distribution orbital inclinations?
"""

import torch
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import json
import warnings

from h_gnn.models import HGNN, TwoBodyHamiltonian, GraphBuilder
from h_gnn.models.baselines import VanillaGNN, create_baseline
from h_gnn.integrators import StormerVerlet
from h_gnn.data import SyntheticCatalog
from h_gnn.eval import MetricsComputer, PhysicsMetrics


def run_rare_regime_experiment(
    config: Dict[str, Any],
    model: HGNN,
    save_dir: str = "results/rare_regime",
) -> Dict[str, Any]:
    """
    Run complete rare-regime evaluation.
    
    Args:
        config: Experiment configuration
        model: Trained H-GNN model
        save_dir: Directory to save results
        
    Returns:
        Dictionary of results
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Config
    mu = config.get("mu", 398600.4418)
    dt = config.get("dt", 60.0)
    n_steps = config.get("eval_steps", 100)
    n_objects = config.get("n_objects", 50)
    
    # Ground truth
    gt_hamiltonian = TwoBodyHamiltonian(mu=mu)
    gt_integrator = StormerVerlet(dt=dt)
    metrics_computer = MetricsComputer(mu=mu)
    
    # Generate test sets
    catalog = SyntheticCatalog(mu=mu, seed=123)
    
    # Inclination bins for systematic evaluation
    inclination_bins = [
        (0, 10, "equatorial"),
        (10, 30, "low"),
        (30, 50, "medium"),
        (50, 70, "high"),
        (70, 90, "polar"),
    ]
    
    results = {
        "hgnn": {},
        "pure_twobody": {},
        "vanilla_gnn": {},
    }
    
    print("Running rare-regime experiment...")
    print("=" * 60)
    
    for inc_min, inc_max, regime_name in inclination_bins:
        print(f"\nEvaluating {regime_name} regime ({inc_min}-{inc_max}°)...")
        
        # Generate test data for this inclination range
        q_test, p_test = catalog.generate_training_set(
            n_objects=n_objects,
            inclination_range=(np.radians(inc_min), np.radians(inc_max)),
        )
        
        # Ground truth trajectory
        with torch.no_grad():
            q_gt, p_gt = gt_integrator.integrate(
                q_test, p_test, gt_hamiltonian, n_steps, store_trajectory=True
            )
        
        # Evaluate H-GNN
        model.eval()
        model.enable_interaction()
        
        with torch.no_grad():
            q_pred, p_pred = model(q_test, p_test, n_steps=n_steps, return_trajectory=True)
        
        error_hgnn = torch.mean((q_pred[-1] - q_gt[-1]) ** 2).item()
        physics_hgnn = metrics_computer.compute_physics_metrics(q_pred, p_pred)
        
        results["hgnn"][regime_name] = {
            "mse": error_hgnn,
            "energy_drift": physics_hgnn.energy_drift_mean,
            "L_drift": physics_hgnn.angular_momentum_drift_mean,
        }
        
        # Evaluate pure two-body (λ=0)
        model.disable_interaction()
        
        with torch.no_grad():
            q_pred_0, p_pred_0 = model(q_test, p_test, n_steps=n_steps, return_trajectory=True)
        
        error_0 = torch.mean((q_pred_0[-1] - q_gt[-1]) ** 2).item()
        physics_0 = metrics_computer.compute_physics_metrics(q_pred_0, p_pred_0)
        
        results["pure_twobody"][regime_name] = {
            "mse": error_0,
            "energy_drift": physics_0.energy_drift_mean,
            "L_drift": physics_0.angular_momentum_drift_mean,
        }
        
        model.enable_interaction()
        
        # Evaluate Vanilla GNN
        try:
            gnn = VanillaGNN(hidden_dim=64, n_layers=2)
            graph_builder = GraphBuilder(radius=1000.0)
            graph = graph_builder(q_test)
            
            with torch.no_grad():
                q_pred_gnn, _ = gnn(q_test, p_test, graph.edge_index, n_steps=n_steps, return_trajectory=True)
            
            error_gnn = torch.mean((q_pred_gnn[-1] - q_gt[-1]) ** 2).item()
            
            results["vanilla_gnn"][regime_name] = {
                "mse": error_gnn,
            }
        except Exception as e:
            results["vanilla_gnn"][regime_name] = {"mse": float('nan'), "error": str(e)}
        
        print(f"  H-GNN MSE: {error_hgnn:.6f}")
        print(f"  Two-body MSE: {error_0:.6f}")
    
    # Compute generalization metrics
    print("\n" + "=" * 60)
    print("Generalization Analysis")
    print("=" * 60)
    
    for model_name in results:
        if "equatorial" in results[model_name] and "polar" in results[model_name]:
            eq_mse = results[model_name]["equatorial"]["mse"]
            polar_mse = results[model_name]["polar"]["mse"]
            
            if eq_mse > 0:
                gen_gap = polar_mse / eq_mse
                results[model_name]["generalization_gap"] = gen_gap
                print(f"{model_name}: generalization gap = {gen_gap:.2f}x")
    
    # Save results
    results_file = Path(save_dir) / "rare_regime_results.json"
    
    # Convert numpy/torch to Python types for JSON
    def convert_for_json(obj):
        if isinstance(obj, (np.floating, torch.Tensor)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def compute_improvement_metrics(results: Dict) -> Dict[str, float]:
    """Compute key improvement metrics."""
    metrics = {}
    
    # H-GNN vs Vanilla GNN improvement
    if "hgnn" in results and "vanilla_gnn" in results:
        if "generalization_gap" in results["hgnn"] and "generalization_gap" in results["vanilla_gnn"]:
            hgnn_gap = results["hgnn"]["generalization_gap"]
            gnn_gap = results["vanilla_gnn"]["generalization_gap"]
            
            if gnn_gap > 0:
                improvement = (gnn_gap - hgnn_gap) / gnn_gap * 100
                metrics["improvement_over_gnn"] = improvement
    
    # H-GNN vs pure two-body
    if "hgnn" in results and "pure_twobody" in results:
        if "polar" in results["hgnn"] and "polar" in results["pure_twobody"]:
            hgnn_polar = results["hgnn"]["polar"]["mse"]
            twobody_polar = results["pure_twobody"]["polar"]["mse"]
            
            if twobody_polar > 0:
                improvement = (twobody_polar - hgnn_polar) / twobody_polar * 100
                metrics["improvement_over_twobody_polar"] = improvement
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="hgnn_model.pt")
    parser.add_argument("--n-objects", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=100)
    args = parser.parse_args()
    
    config = {
        "mu": 398600.4418,
        "dt": 60.0,
        "eval_steps": args.n_steps,
        "n_objects": args.n_objects,
    }
    
    # Load model
    model = HGNN(
        mu=config["mu"],
        interaction_strength=0.01,
        interaction_radius=1000.0,
        hidden_dims=[64, 64, 32],
        dt=config["dt"],
    ).double()  # Convert to float64
    
    if Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, weights_only=True), strict=False)
        print(f"Loaded model from {args.model_path}")
    else:
        print("Warning: Model not found, using random initialization")
    
    results = run_rare_regime_experiment(config, model)
    improvements = compute_improvement_metrics(results)
    
    print("\nKey Metrics:")
    for k, v in improvements.items():
        print(f"  {k}: {v:.1f}%")
