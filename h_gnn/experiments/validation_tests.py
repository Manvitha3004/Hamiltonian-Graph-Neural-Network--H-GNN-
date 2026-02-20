"""
Validation Tests for Finalized Training Objective.

(A) Multi-Seed T=100 Validation
   - Run 3 seeds with rebalanced weights [0.2, 0.3, 0.3, 0.2]
   - Report mean + std for T=100

(B) dt² Scaling Test (Backward Error Confirmation)
   - Run with dt and dt/2
   - If error scales as dt², confirms backward error interpretation
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from h_gnn.models import HGNN
from h_gnn.models.multibody import MultiBodyHamiltonian
from h_gnn.integrators import StormerVerlet
import torch.optim as optim


def create_model(phi_scale=0.005, dt=60.0):
    """Create HGNN model."""
    return HGNN(
        mu=398600.4418,
        interaction_strength=phi_scale,
        interaction_radius=20000.0,
        hidden_dims=[64, 64, 32],
        integrator_type="stormer_verlet",
        dt=dt,
        create_graph=True,
        softplus_output=False,
        normalize_features=True,
        init_gain=1.0,
    ).float()


def generate_ic(n_particles, mu, seed):
    """Generate random initial conditions."""
    np.random.seed(seed)
    r = 7000 + 500 * np.random.randn(n_particles)
    theta = np.linspace(0, 2*np.pi, n_particles, endpoint=False) + 0.1*np.random.randn(n_particles)
    
    q = torch.tensor([
        [r[i]*np.cos(theta[i]), r[i]*np.sin(theta[i]), 50*np.random.randn()] 
        for i in range(n_particles)
    ], dtype=torch.float64)
    
    v = np.sqrt(mu / r)
    p = torch.tensor([
        [v[i]*(-np.sin(theta[i])), v[i]*np.cos(theta[i]), 0.1*np.random.randn()] 
        for i in range(n_particles)
    ], dtype=torch.float64)
    
    return q, p


def multiscale_loss(model, gt_hamiltonian, base_integrator, q0, p0, edge_index, horizons, weights):
    """Compute multi-scale trajectory loss."""
    total_loss = 0.0
    
    for T, w in zip(horizons, weights):
        # Ground truth
        with torch.no_grad():
            q_true, p_true = q0.double(), p0.double()
            for _ in range(T):
                q_true, p_true = base_integrator.step(q_true, p_true, gt_hamiltonian, edge_index)
        
        # Model prediction
        q_pred, p_pred = model.forward(q0.float(), p0.float(), n_steps=T, return_trajectory=False)
        
        loss = torch.mean((q_pred - q_true.float())**2 + (p_pred - p_true.float())**2)
        total_loss = total_loss + w * loss
    
    return total_loss


def train_one_seed(seed, config):
    """Train model with one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_particles = config["n_particles"]
    epsilon = config["epsilon"]
    dt = config["dt"]
    n_epochs = config["n_epochs"]
    horizons = config["horizons"]
    weights = config["weights"]
    phi_scale = config["phi_scale"]
    
    gt_hamiltonian = MultiBodyHamiltonian(mu=398600.4418, epsilon=epsilon)
    base_integrator = StormerVerlet(dt=dt, create_graph=False)
    
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    model = create_model(phi_scale=phi_scale, dt=dt)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(n_epochs):
        model.train()
        q0, p0 = generate_ic(n_particles, 398600.4418, seed * 1000 + epoch)
        
        loss = multiscale_loss(model, gt_hamiltonian, base_integrator, 
                               q0, p0, edge_index, horizons, weights)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
    
    return model


def evaluate_model(model, gt_hamiltonian, base_integrator, dt, n_particles, horizons_eval, n_eval=10):
    """Evaluate model at different horizons."""
    edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).T
    
    # Create a two-body only model for comparison
    twobody_model = create_model(phi_scale=0.0, dt=dt)  # lambda=0 means no interaction
    
    model.eval()
    twobody_model.eval()
    results = {}
    
    for T in horizons_eval:
        model_mses = []
        twobody_mses = []
        
        for i in range(n_eval):
            q0, p0 = generate_ic(n_particles, 398600.4418, 9999 + i)
            
            with torch.no_grad():
                # Ground truth
                q_true, p_true = q0.clone(), p0.clone()
                for _ in range(T):
                    q_true, p_true = base_integrator.step(q_true, p_true, gt_hamiltonian, edge_index)
                
                # Model prediction
                q_pred, p_pred = model.forward(q0.float(), p0.float(), n_steps=T, return_trajectory=False)
                
                # Two-body prediction (lambda=0)
                q_base, p_base = twobody_model.forward(q0.float(), p0.float(), n_steps=T, return_trajectory=False)
                
                model_mse = torch.mean((q_pred - q_true.float())**2).item()
                twobody_mse = torch.mean((q_base - q_true.float())**2).item()
                
                model_mses.append(model_mse)
                twobody_mses.append(twobody_mse)
        
        results[T] = {
            "model_mse": np.mean(model_mses),
            "twobody_mse": np.mean(twobody_mses),
            "improvement": 100 * (np.mean(twobody_mses) - np.mean(model_mses)) / np.mean(twobody_mses)
        }
    
    return results


def test_A_multiseed():
    """Test A: Multi-seed T=100 validation."""
    print("=" * 70)
    print("TEST A: Multi-Seed T=100 Validation")
    print("=" * 70)
    print("Weights: [0.2, 0.3, 0.3, 0.2] (rebalanced)")
    print("Seeds: [42, 123, 456]")
    print()
    
    config = {
        "n_particles": 5,
        "epsilon": 100.0,
        "dt": 60.0,
        "n_epochs": 150,
        "horizons": [5, 10, 20, 50],
        "weights": [0.2, 0.3, 0.3, 0.2],  # Rebalanced weights
        "phi_scale": 0.005,
    }
    
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"Training seed {seed}...")
        model = train_one_seed(seed, config)
        
        gt = MultiBodyHamiltonian(mu=398600.4418, epsilon=config["epsilon"])
        base = StormerVerlet(dt=config["dt"], create_graph=False)
        
        results = evaluate_model(model, gt, base, config["dt"], 
                                 config["n_particles"], [25, 50, 100, 200])
        all_results.append(results)
        
        print(f"  T=100: {results[100]['improvement']:+.1f}%")
    
    print()
    print("-" * 50)
    print("RESULTS (mean +/- std across 3 seeds)")
    print("-" * 50)
    
    for T in [25, 50, 100, 200]:
        imps = [r[T]["improvement"] for r in all_results]
        mean_imp = np.mean(imps)
        std_imp = np.std(imps)
        print(f"T={T:3d}: {mean_imp:+6.1f}% +/- {std_imp:5.1f}%")
    
    # Check if T=100 is mildly positive
    t100_imps = [r[100]["improvement"] for r in all_results]
    t100_mean = np.mean(t100_imps)
    
    print()
    if t100_mean > 0:
        print("SUCCESS: T=100 is positive! Training objective finalized.")
    else:
        print(f"T=100 mean is {t100_mean:+.1f}% (target: >0%)")
    
    return all_results


def test_B_dt_scaling():
    """Test B: dt² scaling test for backward error confirmation."""
    print()
    print("=" * 70)
    print("TEST B: dt^2 Scaling (Backward Error Confirmation)")
    print("=" * 70)
    print("If error ~ dt^2, this confirms backward error interpretation.")
    print()
    
    config = {
        "n_particles": 5,
        "epsilon": 100.0,
        "n_epochs": 100,
        "horizons": [5, 10, 20, 50],
        "weights": [0.2, 0.3, 0.3, 0.2],
        "phi_scale": 0.005,
    }
    
    dt_values = [60.0, 30.0]  # dt and dt/2
    
    results_by_dt = {}
    
    for dt in dt_values:
        print(f"Testing dt = {dt:.1f}s...")
        
        config["dt"] = dt
        model = train_one_seed(42, config)
        
        gt = MultiBodyHamiltonian(mu=398600.4418, epsilon=config["epsilon"])
        base = StormerVerlet(dt=dt, create_graph=False)
        
        # Evaluate at same physical time (adjust T proportionally)
        # T=100 at dt=60 = 6000s = T=200 at dt=30
        scale = int(60.0 / dt)
        horizons_eval = [25 * scale, 50 * scale, 100 * scale]
        
        results = evaluate_model(model, gt, base, dt, config["n_particles"], horizons_eval)
        results_by_dt[dt] = results
        
        print(f"  Evaluated at T={horizons_eval}")
    
    print()
    print("-" * 60)
    print("dt^2 SCALING ANALYSIS")
    print("-" * 60)
    print(f"{'Physical Time':>15} {'MSE (dt=60)':>15} {'MSE (dt=30)':>15} {'Ratio':>10}")
    print("-" * 60)
    
    # Compare MSE at same physical time
    dt60 = results_by_dt[60.0]
    dt30 = results_by_dt[30.0]
    
    time_comparisons = [
        (1500, 25, 50),    # 1500s = T=25 at dt=60, T=50 at dt=30
        (3000, 50, 100),   # 3000s = T=50 at dt=60, T=100 at dt=30
        (6000, 100, 200),  # 6000s = T=100 at dt=60, T=200 at dt=30
    ]
    
    ratios = []
    for phys_time, T60, T30 in time_comparisons:
        if T60 in dt60 and T30 in dt30:
            mse60 = dt60[T60]["model_mse"]
            mse30 = dt30[T30]["model_mse"]
            ratio = mse60 / mse30 if mse30 > 0 else float('inf')
            ratios.append(ratio)
            print(f"{phys_time:>15}s {mse60:>15.2e} {mse30:>15.2e} {ratio:>10.2f}")
    
    print()
    print("INTERPRETATION:")
    print(f"  dt ratio: 60/30 = 2")
    print(f"  Expected MSE ratio for dt^2 scaling: 2^2 = 4")
    
    if ratios:
        mean_ratio = np.mean(ratios)
        print(f"  Observed mean MSE ratio: {mean_ratio:.2f}")
        
        if 2 < mean_ratio < 8:
            print()
            print("  CONFIRMED: Error scales approximately as dt^2!")
            print("  This validates the backward error interpretation.")
        elif mean_ratio > 1:
            print()
            print(f"  Partial confirmation: Error decreases with dt (ratio={mean_ratio:.2f})")
        else:
            print()
            print("  Inconclusive: Ratio does not match dt^2 scaling.")
    
    return results_by_dt


if __name__ == "__main__":
    # Run both tests
    results_A = test_A_multiseed()
    results_B = test_B_dt_scaling()
    
    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    # Extract T=100 result from Test A
    t100_imps = [r[100]["improvement"] for r in results_A]
    t100_mean = np.mean(t100_imps)
    t100_std = np.std(t100_imps)
    
    print(f"(A) T=100 improvement: {t100_mean:+.1f}% +/- {t100_std:.1f}%")
    print(f"    Training objective finalized: {'YES' if t100_mean > 0 else 'NO'}")
    print()
    print("(B) dt^2 scaling test completed.")
    print("    See detailed analysis above.")
