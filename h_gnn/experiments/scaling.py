"""
Scaling Analysis Experiment.

Measures runtime and memory scaling as number of objects increases.
Target: O(N log N) complexity via FAISS neighbor search.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any
from pathlib import Path
import json
import gc

from h_gnn.models import HGNN, TwoBodyHamiltonian, GraphBuilder
from h_gnn.models.baselines import VanillaGNN
from h_gnn.integrators import StormerVerlet
from h_gnn.data import SyntheticCatalog


def measure_runtime(
    model: torch.nn.Module,
    q: torch.Tensor,
    p: torch.Tensor,
    n_steps: int = 10,
    n_warmup: int = 2,
    n_trials: int = 5,
    **kwargs,
) -> Dict[str, float]:
    """
    Measure runtime for a model.
    
    Args:
        model: Model to benchmark
        q: Positions
        p: Momenta
        n_steps: Integration steps
        n_warmup: Warmup iterations
        n_trials: Measurement trials
        **kwargs: Additional args for model
        
    Returns:
        Dict with mean, std, min, max runtime in ms
    """
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(q, p, n_steps=n_steps, **kwargs)
    
    # Measurement
    times = []
    for _ in range(n_trials):
        gc.collect()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(q, p, n_steps=n_steps, **kwargs)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def measure_memory(
    model: torch.nn.Module,
    q: torch.Tensor,
    p: torch.Tensor,
    n_steps: int = 10,
    **kwargs,
) -> Dict[str, float]:
    """
    Measure memory usage.
    
    Note: Accurate GPU memory measurement requires CUDA.
    For CPU, this is approximate.
    """
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            _ = model(q, p, n_steps=n_steps, **kwargs)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
    else:
        # Approximate CPU memory (less accurate)
        import tracemalloc
        tracemalloc.start()
        
        with torch.no_grad():
            _ = model(q, p, n_steps=n_steps, **kwargs)
        
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory = peak_memory / 1e6  # MB
    
    return {
        "peak_memory_mb": peak_memory,
    }


def run_scaling_experiment(
    config: Dict[str, Any],
    n_objects_list: List[int] = [50, 100, 200, 500, 1000],
    save_dir: str = "results/scaling",
) -> Dict[str, Any]:
    """
    Run scaling analysis across different N values.
    
    Args:
        config: Configuration
        n_objects_list: List of N values to test
        save_dir: Directory for results
        
    Returns:
        Scaling results dictionary
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    mu = config.get("mu", 398600.4418)
    dt = config.get("dt", 60.0)
    n_steps = config.get("n_steps", 10)
    
    catalog = SyntheticCatalog(mu=mu, seed=42)
    
    results = {
        "hgnn": {"runtime": [], "memory": [], "n_objects": []},
        "pure_twobody": {"runtime": [], "memory": [], "n_objects": []},
        "graph_construction": {"runtime": [], "n_objects": []},
    }
    
    print("Running scaling experiment...")
    print("=" * 60)
    
    for N in n_objects_list:
        print(f"\nN = {N} objects...")
        
        # Generate data
        q, p = catalog.generate_training_set(n_objects=N)
        
        # H-GNN
        model = HGNN(
            mu=mu,
            interaction_strength=0.01,
            interaction_radius=1000.0,
            hidden_dims=[64, 64, 32],
            dt=dt,
        ).double()  # Convert to float64
        
        runtime = measure_runtime(model, q, p, n_steps=n_steps)
        memory = measure_memory(model, q, p, n_steps=n_steps)
        
        results["hgnn"]["runtime"].append(runtime["mean_ms"])
        results["hgnn"]["memory"].append(memory["peak_memory_mb"])
        results["hgnn"]["n_objects"].append(N)
        
        print(f"  H-GNN: {runtime['mean_ms']:.2f} ms, {memory['peak_memory_mb']:.2f} MB")
        
        # Pure two-body (λ=0)
        model.disable_interaction()
        
        runtime_0 = measure_runtime(model, q, p, n_steps=n_steps)
        memory_0 = measure_memory(model, q, p, n_steps=n_steps)
        
        results["pure_twobody"]["runtime"].append(runtime_0["mean_ms"])
        results["pure_twobody"]["memory"].append(memory_0["peak_memory_mb"])
        results["pure_twobody"]["n_objects"].append(N)
        
        print(f"  Two-body: {runtime_0['mean_ms']:.2f} ms, {memory_0['peak_memory_mb']:.2f} MB")
        
        # Graph construction only
        graph_builder = GraphBuilder(radius=1000.0, use_faiss=True)
        
        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = graph_builder(q)
            times.append((time.perf_counter() - start) * 1000)
        
        results["graph_construction"]["runtime"].append(np.mean(times))
        results["graph_construction"]["n_objects"].append(N)
        
        print(f"  Graph: {np.mean(times):.2f} ms")
        
        # Cleanup
        del model
        gc.collect()
    
    # Fit scaling exponents
    print("\n" + "=" * 60)
    print("Scaling Analysis")
    print("=" * 60)
    
    for model_name in ["hgnn", "pure_twobody", "graph_construction"]:
        n_arr = np.array(results[model_name]["n_objects"])
        t_arr = np.array(results[model_name]["runtime"])
        
        # Fit log(t) = α * log(N) + β => t = c * N^α
        if len(n_arr) >= 2:
            log_n = np.log(n_arr)
            log_t = np.log(t_arr)
            
            # Linear regression
            A = np.vstack([log_n, np.ones(len(log_n))]).T
            alpha, beta = np.linalg.lstsq(A, log_t, rcond=None)[0]
            
            results[model_name]["scaling_exponent"] = alpha
            print(f"{model_name}: O(N^{alpha:.2f})")
    
    # Save results
    results_file = Path(save_dir) / "scaling_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Runtime
        ax = axes[0]
        for model_name in ["hgnn", "pure_twobody", "graph_construction"]:
            n = results[model_name]["n_objects"]
            t = results[model_name]["runtime"]
            ax.loglog(n, t, 'o-', label=model_name)
        
        ax.set_xlabel("Number of objects (N)")
        ax.set_ylabel("Runtime (ms)")
        ax.set_title("Runtime Scaling")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Memory
        ax = axes[1]
        for model_name in ["hgnn", "pure_twobody"]:
            n = results[model_name]["n_objects"]
            m = results[model_name]["memory"]
            ax.loglog(n, m, 'o-', label=model_name)
        
        ax.set_xlabel("Number of objects (N)")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory Scaling")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(save_dir) / "scaling_plot.png", dpi=150)
        print(f"Plot saved to: {Path(save_dir) / 'scaling_plot.png'}")
        
    except ImportError:
        print("matplotlib not available for plotting")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=1000)
    args = parser.parse_args()
    
    config = {
        "mu": 398600.4418,
        "dt": 60.0,
        "n_steps": 10,
    }
    
    n_objects_list = [50, 100, 200, 500]
    if args.max_n >= 1000:
        n_objects_list.append(1000)
    if args.max_n >= 2000:
        n_objects_list.append(2000)
    if args.max_n >= 5000:
        n_objects_list.append(5000)
    
    results = run_scaling_experiment(config, n_objects_list)
