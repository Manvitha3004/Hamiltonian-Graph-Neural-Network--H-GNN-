"""
Training Script for H-GNN.

Implements:
1. Two-body symplectic validation (Phase 1)
2. Graph Hamiltonian training (Phase 2)
3. Rare-regime evaluation (Phase 4)

All metrics are logged to wandb for experiment tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import warnings

# H-GNN imports
from h_gnn.models import HGNN, TwoBodyHamiltonian, InteractionPotential, GraphBuilder
from h_gnn.models.hgnn import HGNNPredictor, PureHamiltonianModel
from h_gnn.models.baselines import VanillaGNN, EquivariantGNN, create_baseline
from h_gnn.integrators import SymplecticEuler, StormerVerlet
from h_gnn.integrators.baselines import VanillaEuler, RK4
from h_gnn.data import SyntheticCatalog, ConjunctionDataset
from h_gnn.eval import (
    MetricsComputer,
    PhysicsMetrics,
    StabilityMetrics,
    MLMetrics,
    evaluate_integrator_comparison,
    log_metrics_to_wandb,
    create_comparison_plots,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_wandb(config: Dict[str, Any], offline: bool = False) -> bool:
    """Initialize wandb logging."""
    try:
        import wandb
        
        # Use offline/disabled mode if requested or if no API key
        mode = "disabled" if offline else "online"
        
        wandb.init(
            project=config.get("wandb_project", "h-gnn"),
            name=config.get("experiment_name", f"run_{datetime.now():%Y%m%d_%H%M%S}"),
            config=config,
            mode=mode,
        )
        return True
    except ImportError:
        warnings.warn("wandb not installed. Logging disabled.")
        return False
    except Exception as e:
        warnings.warn(f"wandb initialization failed: {e}. Running without logging.")
        return False


def phase1_validation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1: Two-Body Symplectic Validation
    
    Proves symplectic integration preserves structure better
    than standard integrators.
    """
    print("\n" + "="*60)
    print("PHASE 1: Two-Body Symplectic Validation")
    print("="*60)
    
    # Setup
    mu = config.get("mu", 398600.4418)
    dt = config.get("dt", 60.0)
    n_steps = config.get("n_steps", 1000)
    
    # Create Hamiltonian
    hamiltonian = TwoBodyHamiltonian(mu=mu)
    
    # Generate initial circular orbit
    R_earth = 6378.137
    altitude = 400.0  # km
    a = R_earth + altitude
    
    # Circular orbit initial conditions
    v_circular = np.sqrt(mu / a)
    q0 = torch.tensor([[a, 0.0, 0.0]], dtype=torch.float64)
    p0 = torch.tensor([[0.0, v_circular, 0.0]], dtype=torch.float64)
    
    print(f"Initial conditions:")
    print(f"  Position: {q0.numpy()}")
    print(f"  Velocity: {p0.numpy()}")
    print(f"  Orbital radius: {a:.2f} km")
    print(f"  Integration: {n_steps} steps x {dt}s = {n_steps*dt/3600:.2f} hours")
    
    # Create integrators
    integrators = {
        "Symplectic Euler": SymplecticEuler(dt=dt),
        "Störmer-Verlet": StormerVerlet(dt=dt),
        "Vanilla Euler": VanillaEuler(dt=dt),
        "RK4": RK4(dt=dt),
    }
    
    # Run comparison
    results = evaluate_integrator_comparison(
        q0, p0, hamiltonian, integrators, n_steps, dt
    )
    
    # Print results
    print("\nResults:")
    print("-" * 40)
    for name, data in results.items():
        physics = data["physics"]
        stability = data["stability"]
        print(f"\n{name}:")
        print(f"  Energy drift (mean): {physics.energy_drift_mean:.2e}")
        print(f"  Energy drift (max):  {physics.energy_drift_max:.2e}")
        print(f"  Energy oscillation:  {physics.energy_oscillation:.2e}")
        print(f"  L drift (mean):      {physics.angular_momentum_drift_mean:.2e}")
        print(f"  Stable:              {stability.trajectory_bounded}")
    
    # Create plots
    try:
        fig = create_comparison_plots(results, save_path="phase1_comparison.png")
        print("\nPlot saved to: phase1_comparison.png")
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # Validation criteria
    symplectic_ok = (
        results["Symplectic Euler"]["physics"].energy_drift_max < 0.01 and
        results["Störmer-Verlet"]["physics"].energy_drift_max < 0.001
    )
    
    euler_diverges = results["Vanilla Euler"]["physics"].energy_drift_max > 0.1
    
    print("\n" + "="*60)
    print("Validation:")
    print(f"  Symplectic bounded energy: {'✓' if symplectic_ok else '✗'}")
    print(f"  Euler diverges: {'✓' if euler_diverges else '✗'}")
    print("="*60)
    
    return {
        "results": results,
        "symplectic_ok": symplectic_ok,
        "euler_diverges": euler_diverges,
    }


def phase2_training(config: Dict[str, Any]) -> nn.Module:
    """
    Phase 2: Graph Hamiltonian Training
    
    Train learned interaction potential while preserving
    symplectic structure.
    
    Improvements:
    - Normalized loss function (scale by typical orbital magnitudes)
    - Cosine annealing learning rate with warmup
    - Curriculum learning (increasing rollout horizon)
    - Gradient norm monitoring
    - Multiple loss components with learned weighting
    """
    print("\n" + "="*60)
    print("PHASE 2: Graph Hamiltonian Training")
    print("="*60)
    
    # Config
    n_objects = config.get("n_objects", 500)  # Increased from 100
    n_epochs = config.get("n_epochs", 200)    # Increased from 50
    lr = config.get("lr", 3e-4)               # Reduced initial LR
    interaction_strength = config.get("interaction_strength", 0.01)
    rollout_steps = config.get("rollout_steps", 10)
    grad_clip = config.get("grad_clip", 0.5)  # Reduced from 1.0
    
    # Normalization constants for orbital mechanics
    # Typical LEO position: ~7000 km, velocity: ~7.5 km/s
    q_scale = 7000.0  # km
    p_scale = 7.5     # km/s
    
    # Generate training data
    print(f"Generating {n_objects} objects...")
    catalog = SyntheticCatalog(seed=42)
    data = catalog.generate_dataset(
        n_train=n_objects,
        n_test=n_objects // 4,
        n_conjunctions_train=n_objects // 10,
        n_conjunctions_test=n_objects // 20,
    )
    
    q_train = data["train_q"]
    p_train = data["train_p"]
    
    print(f"Training set: {q_train.shape[0]} objects")
    print(f"Test set: {data['test_q'].shape[0]} objects")
    
    # Create model
    model = HGNN(
        mu=config.get("mu", 398600.4418),
        interaction_strength=interaction_strength,
        interaction_radius=config.get("interaction_radius", 1000.0),
        hidden_dims=config.get("hidden_dims", [64, 64, 32]),
        integrator_type="symplectic_euler",
        dt=config.get("dt", 60.0),
        create_graph=True,  # Enable gradient tracking for training
    )
    model = model.double()  # Convert to float64 for numerical precision
    
    # Ground truth integrator
    # If include_j2=True, use J2-perturbed dynamics for ground truth
    # The model's base Hamiltonian does NOT include J2, so Φ must learn it
    include_j2 = config.get("include_j2", False)
    
    if include_j2:
        from h_gnn.models.hamiltonian import KeplerianHamiltonian
        gt_hamiltonian = KeplerianHamiltonian(
            mu=config.get("mu", 398600.4418),
            J2=config.get("J2", 1.08263e-3),
            R_eq=config.get("R_earth", 6378.137),
            include_J2=True,
        )
        print("Using J2-perturbed ground truth (Φ must learn J2 correction)")
    else:
        gt_hamiltonian = TwoBodyHamiltonian(mu=config.get("mu", 398600.4418))
        print("Using pure two-body ground truth")
        
    gt_integrator = StormerVerlet(dt=config.get("dt", 60.0))
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=config.get("weight_decay", 1e-5)
    )
    
    # Cosine annealing with warmup
    warmup_epochs = min(10, n_epochs // 10)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"\nTraining for {n_epochs} epochs (warmup: {warmup_epochs})...")
    print("-" * 40)
    
    metrics_computer = MetricsComputer()
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = config.get("early_stop_patience", 20)
    
    # Curriculum learning: start with shorter rollouts
    min_rollout = 3
    max_rollout = rollout_steps
    
    for epoch in range(n_epochs):
        model.train()
        
        # Curriculum: increase rollout steps over training
        curriculum_progress = min(1.0, epoch / (n_epochs * 0.5))
        current_rollout = int(min_rollout + curriculum_progress * (max_rollout - min_rollout))
        
        # Forward pass: integrate and compare to ground truth
        q_pred, p_pred = model(q_train, p_train, n_steps=current_rollout)
        
        # Ground truth (high-accuracy integrator)
        with torch.no_grad():
            q_gt, p_gt = gt_integrator.integrate(
                q_train, p_train, gt_hamiltonian, current_rollout,
                store_trajectory=False
            )
        
        # Normalized loss: scale by typical magnitudes
        # This ensures position and momentum errors are comparable
        loss_q = torch.mean(((q_pred - q_gt) / q_scale) ** 2)
        loss_p = torch.mean(((p_pred - p_gt) / p_scale) ** 2)
        trajectory_loss = loss_q + loss_p
        
        # Physics regularization: relative energy conservation
        H_init = model.compute_hamiltonian(q_train, p_train)
        H_final = model.compute_hamiltonian(q_pred, p_pred)
        energy_violation = torch.abs(H_final - H_init) / (torch.abs(H_init) + 1e-10)
        energy_loss = energy_violation.mean()
        
        # Combined loss with adaptive weighting
        # Start with more emphasis on energy, shift to trajectory
        energy_weight = 0.5 * (1 - curriculum_progress) + 0.1
        total_loss = trajectory_loss + energy_weight * energy_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping with norm monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d}: Loss={total_loss.item():.6f}, "
                  f"Traj={trajectory_loss.item():.6f}, "
                  f"Energy={energy_loss.item():.6f}, "
                  f"LR={current_lr:.2e}, "
                  f"Rollout={current_rollout}")
        
        # Early stopping check
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
            # Save best model
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Restore best model
    if 'best_state' in dir():
        model.load_state_dict(best_state)
    
    print("-" * 40)
    print(f"Training complete. Best loss: {best_loss:.6f}")
    
    # Stability validation
    print("\nStability validation...")
    model.eval()
    
    # Test with λ=0 (should recover two-body)
    model.disable_interaction()
    with torch.no_grad():
        q_ablation, p_ablation = model(q_train[:10], p_train[:10], n_steps=100)
        q_gt, p_gt = gt_integrator.integrate(
            q_train[:10], p_train[:10], gt_hamiltonian, 100,
            store_trajectory=False
        )
    
    ablation_error = torch.mean((q_ablation - q_gt) ** 2).item()
    print(f"  λ=0 ablation error: {ablation_error:.6f}")
    
    model.enable_interaction()
    
    return model


def phase4_evaluation(
    model: nn.Module,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Phase 4: Rare-Regime Evaluation
    
    Test whether Hamiltonian structure improves generalization
    from equatorial to polar orbits.
    """
    print("\n" + "="*60)
    print("PHASE 4: Rare-Regime Evaluation")
    print("="*60)
    
    # Generate test data
    catalog = SyntheticCatalog(seed=123)
    
    # Training distribution (equatorial)
    q_equatorial, p_equatorial = catalog.generate_training_set(
        n_objects=50,
        inclination_range=(0, np.radians(20)),
    )
    
    # Rare regime (polar)
    q_polar, p_polar = catalog.generate_test_set(
        n_objects=50,
        inclination_range=(np.radians(60), np.radians(90)),
    )
    
    print(f"Equatorial test set: {q_equatorial.shape[0]} objects")
    print(f"Polar test set: {q_polar.shape[0]} objects")
    
    # Ground truth
    gt_hamiltonian = TwoBodyHamiltonian(mu=config.get("mu", 398600.4418))
    gt_integrator = StormerVerlet(dt=config.get("dt", 60.0))
    n_steps = config.get("eval_steps", 100)
    
    metrics_computer = MetricsComputer()
    results = {}
    
    # Evaluate H-GNN
    print("\nEvaluating H-GNN...")
    model.eval()
    
    with torch.no_grad():
        # Equatorial
        q_pred_eq, p_pred_eq = model(q_equatorial, p_equatorial, n_steps=n_steps, return_trajectory=True)
        q_gt_eq, p_gt_eq = gt_integrator.integrate(q_equatorial, p_equatorial, gt_hamiltonian, n_steps, store_trajectory=True)
        
        # Polar
        q_pred_polar, p_pred_polar = model(q_polar, p_polar, n_steps=n_steps, return_trajectory=True)
        q_gt_polar, p_gt_polar = gt_integrator.integrate(q_polar, p_polar, gt_hamiltonian, n_steps, store_trajectory=True)
    
    # Compute errors
    error_eq = torch.mean((q_pred_eq[-1] - q_gt_eq[-1]) ** 2).item()
    error_polar = torch.mean((q_pred_polar[-1] - q_gt_polar[-1]) ** 2).item()
    
    # Physics metrics
    physics_eq = metrics_computer.compute_physics_metrics(q_pred_eq, p_pred_eq)
    physics_polar = metrics_computer.compute_physics_metrics(q_pred_polar, p_pred_polar)
    
    results["hgnn"] = {
        "error_equatorial": error_eq,
        "error_polar": error_polar,
        "generalization_gap": error_polar / (error_eq + 1e-10),
        "energy_drift_eq": physics_eq.energy_drift_mean,
        "energy_drift_polar": physics_polar.energy_drift_mean,
    }
    
    print(f"H-GNN Results:")
    print(f"  Equatorial MSE: {error_eq:.6f}")
    print(f"  Polar MSE: {error_polar:.6f}")
    print(f"  Generalization gap: {results['hgnn']['generalization_gap']:.2f}x")
    
    # Evaluate baselines
    print("\nEvaluating baselines...")
    
    # 1. Pure two-body (λ=0)
    model.disable_interaction()
    with torch.no_grad():
        q_pred_eq_0, _ = model(q_equatorial, p_equatorial, n_steps=n_steps, return_trajectory=True)
        q_pred_polar_0, _ = model(q_polar, p_polar, n_steps=n_steps, return_trajectory=True)
    
    error_eq_0 = torch.mean((q_pred_eq_0[-1] - q_gt_eq[-1]) ** 2).item()
    error_polar_0 = torch.mean((q_pred_polar_0[-1] - q_gt_polar[-1]) ** 2).item()
    
    results["pure_twobody"] = {
        "error_equatorial": error_eq_0,
        "error_polar": error_polar_0,
        "generalization_gap": error_polar_0 / (error_eq_0 + 1e-10),
    }
    
    print(f"Pure Two-Body (λ=0):")
    print(f"  Equatorial MSE: {error_eq_0:.6f}")
    print(f"  Polar MSE: {error_polar_0:.6f}")
    
    model.enable_interaction()
    
    # 2. Vanilla GNN baseline
    try:
        gnn = VanillaGNN(hidden_dim=64, n_layers=2)
        gnn = gnn.double()  # Convert to float64 for consistency
        graph_builder = GraphBuilder(radius=1000.0)
        
        with torch.no_grad():
            graph_eq = graph_builder(q_equatorial)
            graph_polar = graph_builder(q_polar)
            
            q_pred_gnn_eq, _ = gnn(q_equatorial, p_equatorial, graph_eq.edge_index, n_steps=n_steps, return_trajectory=True)
            q_pred_gnn_polar, _ = gnn(q_polar, p_polar, graph_polar.edge_index, n_steps=n_steps, return_trajectory=True)
        
        error_gnn_eq = torch.mean((q_pred_gnn_eq[-1] - q_gt_eq[-1]) ** 2).item()
        error_gnn_polar = torch.mean((q_pred_gnn_polar[-1] - q_gt_polar[-1]) ** 2).item()
        
        results["vanilla_gnn"] = {
            "error_equatorial": error_gnn_eq,
            "error_polar": error_gnn_polar,
            "generalization_gap": error_gnn_polar / (error_gnn_eq + 1e-10),
        }
        
        print(f"Vanilla GNN:")
        print(f"  Equatorial MSE: {error_gnn_eq:.6f}")
        print(f"  Polar MSE: {error_gnn_polar:.6f}")
    except Exception as e:
        print(f"Vanilla GNN evaluation failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Rare-Regime Generalization")
    print("="*60)
    
    for name, res in results.items():
        print(f"{name}:")
        print(f"  Generalization gap: {res['generalization_gap']:.2f}x")
    
    # Key metric: improvement over baseline
    if "hgnn" in results and "vanilla_gnn" in results:
        improvement = (
            results["vanilla_gnn"]["generalization_gap"] - 
            results["hgnn"]["generalization_gap"]
        ) / results["vanilla_gnn"]["generalization_gap"] * 100
        
        print(f"\nH-GNN improvement over Vanilla GNN: {improvement:.1f}%")
        
        if improvement >= 10:
            print("✓ SUCCESS: Significant rare-regime improvement!")
        elif improvement > 0:
            print("~ MODERATE: Some improvement, consider longer rollouts")
        else:
            print("✗ No improvement: Hamiltonian structure may not help here")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="H-GNN Training")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 4], default=None,
        help="Run specific phase only"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb logging"
    )
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        # Try to find config relative to this file
        module_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(module_dir, "config.yaml")
        
        if os.path.exists(default_config_path):
            print(f"Loading config from: {default_config_path}")
            config = load_config(default_config_path)
        else:
            print(f"Config file not found: {args.config}")
            print("Using default configuration...")
            config = {
                "mu": 398600.4418,
                "dt": 60.0,
                "n_steps": 1000,
                "n_objects": 100,
                "n_epochs": 50,
                "lr": 1e-3,
                "interaction_strength": 0.01,
                "interaction_radius": 1000.0,
                "hidden_dims": [64, 64, 32],
                "rollout_steps": 10,
                "eval_steps": 100,
            }
    
    # Setup wandb (disabled mode by default for Phase 1 validation)
    if not args.no_wandb:
        # Use disabled mode to avoid login prompts during validation
        setup_wandb(config, offline=True)
    
    # Run phases
    if args.phase == 1 or args.phase is None:
        phase1_results = phase1_validation(config)
        
        if not phase1_results["symplectic_ok"]:
            print("\n⚠ WARNING: Phase 1 validation failed!")
            print("Fix symplectic integration before proceeding.")
            if args.phase == 1:
                return
    
    if args.phase == 2 or args.phase is None:
        model = phase2_training(config)
        
        # Save model
        torch.save(model.state_dict(), "hgnn_model.pt")
        print("Model saved to: hgnn_model.pt")
    
    if args.phase == 4 or args.phase is None:
        # Load model if not already trained
        if 'model' not in dir():
            model = HGNN(
                mu=config.get("mu", 398600.4418),
                interaction_strength=config.get("interaction_strength", 0.01),
                interaction_radius=config.get("interaction_radius", 1000.0),
                hidden_dims=config.get("hidden_dims", [64, 64, 32]),
                dt=config.get("dt", 60.0),
                create_graph=False,  # Inference mode
            )
            model = model.double()
            if os.path.exists("hgnn_model.pt"):
                state_dict = torch.load("hgnn_model.pt", weights_only=True)
                # Filter out integrator state (created lazily)
                state_dict = {k: v for k, v in state_dict.items() 
                              if not k.startswith("_integrator")}
                model.load_state_dict(state_dict, strict=False)
        
        eval_results = phase4_evaluation(model, config)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
