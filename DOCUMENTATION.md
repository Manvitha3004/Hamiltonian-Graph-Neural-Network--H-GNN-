# Hamiltonian Graph Neural Network (H-GNN)
## Complete Project Documentation

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Core Components](#core-components)
6. [Training Methods](#training-methods)
7. [Experimental Results](#experimental-results)
8. [API Reference](#api-reference)
9. [Configuration](#configuration)
10. [Usage Guide](#usage-guide)

---

## Overview

### What is H-GNN?

The Hamiltonian Graph Neural Network (H-GNN) is a structure-preserving neural network that learns corrections to incomplete Hamiltonians while maintaining symplectic structure. It's designed for orbital mechanics and space situational awareness but applies to any physical system with known base dynamics and unknown interactions.

### Core Innovation

Unlike traditional GNNs that learn dynamics through message passing, H-GNN:
1. **Defines a learnable Hamiltonian** that includes both known physics and learned corrections
2. **Derives dynamics from ∇H** using Hamilton's equations
3. **Uses symplectic integration** to preserve phase space structure
4. **Guarantees bounded energy error** over arbitrary time horizons

### Key Results

| Metric | H-GNN | Vanilla GNN + RK4 |
|--------|-------|-------------------|
| T=50 Improvement | **+20.9%** | -3.0% |
| T=100 Improvement | **+12.0%** | -1.5% |
| T=500 Improvement | **+4.9%** | -1.1% |
| Energy Conservation | Bounded | Unbounded drift |

---

## Theoretical Foundation

### Hamiltonian Mechanics

The Hamiltonian H(q, p) represents total energy:
```
H = T + V = (1/2)|p|² - μ/|q|
```

Hamilton's equations give the dynamics:
```
dq/dt = ∂H/∂p = p
dp/dt = -∂H/∂q = -μq/|q|³
```

### Graph Hamiltonian

For N interacting particles, the total Hamiltonian is:
```
H_G = Σᵢ H_orbit(zᵢ) + λ Σ_{(i,j)∈E} Φ_θ(zᵢ, zⱼ)
```

Where:
- `H_orbit` is the known two-body Keplerian Hamiltonian
- `Φ_θ` is the learned pairwise interaction potential
- `λ` is the interaction strength coupling constant
- `E` is the edge set from graph construction

### Backward Error Analysis

With dt²-structured corrections and learnable α:
```
H = H_base + dt² × α × (1/N) × Σ Φ_ij
```

Where:
- `α` is a **learnable parameter** (initialized to 1e-4)
- `dt²` provides correct scaling for Störmer-Verlet (2nd order integrator)
- `1/N` keeps per-particle force magnitude stable across N

This design:
- Eliminates manual λ tuning (α learns the correct magnitude)
- Ensures integration errors scale as O(dt²) for Störmer-Verlet
- Makes the learned Hamiltonian correspond to the "modified Hamiltonian" in backward error analysis

### Symplectic Integration

Symplectic integrators preserve the symplectic 2-form:
```
ω = Σᵢ dqᵢ ∧ dpᵢ
```

This guarantees:
1. **Phase space volume preservation** (Liouville's theorem)
2. **Bounded oscillatory energy error** (no secular drift)
3. **Exact preservation of quadratic invariants**

---

## Architecture

### Data Flow

```
Input: (q, p) ∈ ℝ^{N×3} × ℝ^{N×3}
         ↓
┌─────────────────────────────┐
│  Graph Construction (FAISS) │
│  radius search O(N log N)   │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Graph Hamiltonian          │
│  H = Σ H_orbit + λ Σ Φ_θ    │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Gradients via Autograd     │
│  ∂H/∂q, ∂H/∂p               │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Symplectic Integration     │
│  Störmer-Verlet (2nd order) │
└─────────────────────────────┘
         ↓
Output: Future states (q_T, p_T)
```

### Interaction Potential Φ_θ

The learned interaction uses SO(3) invariant features:
```
Features = [r, r², ||Δp||, Δq·Δp, Δq·p_avg]
```

Where:
- `r = ||q_j - q_i||` (separation distance)
- `Δp = p_j - p_i` (relative momentum)
- `p_avg = (p_i + p_j)/2` (average momentum)

Architecture:
```
Invariant Features → MLP [64, 64, 32] → Scalar Output
```

---

## Project Structure

```
h_gnn/
├── config.yaml                 # Default configuration
├── train.py                    # Main training entry point
├── eval.py                     # Evaluation utilities
│
├── data/
│   ├── synthetic_catalog.py    # Orbital element generation
│   └── tle_parser.py           # TLE parsing and SGP4
│
├── models/
│   ├── hamiltonian.py          # TwoBodyHamiltonian, GraphHamiltonian
│   ├── interaction.py          # InteractionPotential, ScalableInteraction
│   ├── graph.py                # GraphBuilder, SparseGraph (FAISS)
│   ├── hgnn.py                 # Main HGNN model class
│   ├── multibody.py            # MultiBodyHamiltonian (ground truth)
│   └── baselines.py            # VanillaGNN for comparison
│
├── integrators/
│   ├── symplectic.py           # SymplecticEuler, StormerVerlet
│   └── baselines.py            # VanillaEuler, RK4
│
└── experiments/
    ├── backward_error_training.py  # dt²-structured training
    ├── multiscale_training.py      # Multi-horizon training
    ├── baseline_comparison.py      # vs Vanilla GNN
    ├── epsilon_sweep.py            # ε scaling analysis
    ├── scaling_study.py            # N-particle scaling
    ├── validation_tests.py         # T=100 multi-seed tests
    └── hamiltonian_error_analysis.py  # H accuracy diagnostic
```

---

## Core Components

### 1. TwoBodyHamiltonian

The base Keplerian Hamiltonian for orbital mechanics.

```python
from h_gnn.models.hamiltonian import TwoBodyHamiltonian

H_orbit = TwoBodyHamiltonian(mu=398600.4418)  # Earth's GM

# Compute energy
H = H_orbit(q, p, sum_over_particles=True)

# Compute gradients for dynamics
grad_q, grad_p = H_orbit.compute_gradients(q, p)
```

**Parameters:**
- `mu`: Gravitational parameter (km³/s²). Default: Earth's μ = 398600.4418

### 2. GraphHamiltonian

Combines base Hamiltonian with learned interaction.

```python
from h_gnn.models.hamiltonian import GraphHamiltonian

H_graph = GraphHamiltonian(
    orbit_hamiltonian=H_orbit,
    interaction_potential=Phi,
    interaction_strength=0.01,  # Legacy, ignored if use_dt_scaling=True
    use_dt_scaling=True,        # dt²-structured correction
    alpha_init=1e-4,            # Initial value for learnable α
)

# Compute with graph
H_total = H_graph(q, p, edge_index, dt=60.0)

# Access learned alpha
print(f"Learned alpha: {H_graph.alpha.item()}")
```

**Parameters:**
- `orbit_hamiltonian`: Base TwoBodyHamiltonian
- `interaction_potential`: Learned Φ_θ network
- `interaction_strength`: λ coupling constant (legacy, ignored when use_dt_scaling=True)
- `use_dt_scaling`: If True, uses `H = H_base + dt² × α × (1/N) × Σ Φ`
- `alpha_init`: Initial value for learnable α (default: 1e-4)

### 3. InteractionPotential

Learned pairwise interaction Φ_θ(zᵢ, zⱼ).

```python
from h_gnn.models.interaction import InteractionPotential

Phi = InteractionPotential(
    hidden_dims=[64, 64, 32],
    activation="silu",
    use_residual=True,
    softplus_output=False,   # Allow signed corrections
    normalize_features=True, # Normalize input features
    init_gain=1.0,          # Initial weight magnitude
)

# Compute interaction for pairs
phi_ij = Phi(q_i, p_i, q_j, p_j)  # Shape: (num_edges,)
```

### 4. GraphBuilder

Efficient neighbor search using FAISS.

```python
from h_gnn.models.graph import GraphBuilder

builder = GraphBuilder(
    radius=20000.0,    # Interaction radius (km)
    max_neighbors=50,  # Cap on neighbors per node
    use_faiss=True,    # Use FAISS for O(N log N) complexity
)

graph = builder(q, batch=None)
edge_index = graph.edge_index  # Shape: (2, num_edges)
```

### 5. StormerVerlet Integrator

Second-order symplectic integrator.

```python
from h_gnn.integrators.symplectic import StormerVerlet

integrator = StormerVerlet(
    dt=60.0,           # Time step (seconds)
    create_graph=True, # Enable backprop through integration
)

# Single step
q_new, p_new = integrator.step(q, p, hamiltonian, edge_index)

# Multi-step integration
q_traj, p_traj = integrator.integrate(
    q0, p0, hamiltonian, n_steps=100,
    edge_index=edge_index, store_trajectory=True
)
```

### 6. HGNN Model

Complete model combining all components.

```python
from h_gnn.models.hgnn import HGNN

model = HGNN(
    mu=398600.4418,
    interaction_strength=0.01,
    interaction_radius=20000.0,
    hidden_dims=[64, 64, 32],
    integrator_type="stormer_verlet",
    dt=60.0,
    create_graph=True,
    softplus_output=False,
    normalize_features=True,
    init_gain=1.0,
    use_dt_scaling=True,
)

# Forward pass
q_final, p_final = model(q0, p0, n_steps=50)

# Get trajectory
q_traj, p_traj = model(q0, p0, n_steps=50, return_trajectory=True)
```

---

## Training Methods

### 1. Multi-Scale Training

Train with multiple horizons to balance short and long-term accuracy.

```python
from h_gnn.experiments.multiscale_training import train_multiscale

config = {
    "horizons": [5, 10, 20, 50],
    "weights": [0.2, 0.3, 0.3, 0.2],  # Emphasis on T=10-50
    "lambda_energy": 0.1,             # Energy regularization
    "n_epochs": 300,
    "lr": 1e-4,
}

model = train_multiscale(config)
```

**Loss Function:**
```
L = Σ wₜ × MSE(q_pred(t), q_true(t)) + λ_E × (H_model - H_true)²
```

### 2. Backward-Error Aligned Training

Uses dt²-structured Φ for theoretically motivated scaling.

```python
from h_gnn.experiments.backward_error_training import train_backward_error_aligned

config = {
    "dt_primary": 60.0,
    "dt_secondary": 30.0,
    "use_dual_dt": True,     # Train with both timesteps
    "lambda_drift": 0.1,     # Energy drift regularizer
    "use_dt_scaling": True,  # Enable dt² structure
}

model = train_backward_error_aligned(config)
```

**Key Components:**

1. **dt² Scaling**: `H = H_base + dt² × (1/N) × Σ Φ_tilde`
2. **1/N Normalization**: Keeps per-particle forces stable across N
3. **Energy Drift Regularizer**: `(H(q_T, p_T) - H(q_0, p_0))²`
4. **Dual-dt Training**: Train with dt=60 and dt=30 for dt-robustness

### 3. Curriculum Training

Random horizon sampling for extrapolation.

```python
from h_gnn.experiments.curriculum_training import train_with_curriculum

config = {
    "min_horizon": 10,
    "max_horizon": 200,
    "n_epochs": 300,
}

model = train_with_curriculum(config)
```

---

## Experimental Results

### Learnability Threshold (ε-Sweep)

| ε | Improvement@T=50 | Rare Regime |
|---|------------------|-------------|
| 1 | -1.9% | 12.2% |
| 5 | 15.0% | 23.8% |
| **25** | **23.4%** | 25.5% |
| **50** | **20.9%** | **34.7%** |
| 100 | 13.4% | 17.9% |

**Key Finding**: Optimal ε ≈ 25-50. Below ε≈5, signal too weak.

### Symplectic vs Non-Symplectic

| Horizon | H-GNN (Symplectic) | Vanilla GNN (RK4) |
|---------|--------------------|--------------------|
| T=50 | **+20.9%** | -3.0% |
| T=100 | **+12.0%** | -1.5% |
| T=500 | **+4.9%** | -1.1% |

**Key Finding**: Symplectic structure is essential. Vanilla GNN performs WORSE than physics-only baseline.

### Multi-Seed T=100 Validation

| Metric | Value |
|--------|-------|
| Mean Improvement | **+7.8%** |
| Std | 23.9% |
| All Positive | No (2/3 positive) |

**Status**: Training objective finalized. High variance suggests sensitivity to initialization.

### Hamiltonian Error Analysis

| Energy Scale | Relative Error |
|--------------|----------------|
| 0.3x nominal | 0.02% |
| 1.0x nominal | 0.04% |
| 2.5x nominal | 0.01% |

**Key Finding**: H_model ≈ H_true to <1% error. Long-horizon degradation comes from integration error compounding, not Hamiltonian mismatch.

---

## API Reference

### HGNN Constructor

```python
HGNN(
    mu: float = 398600.4418,           # Gravitational parameter
    interaction_strength: float = 0.01, # λ coupling constant (legacy, ignored if use_dt_scaling=True)
    interaction_radius: float = 1000.0, # Neighbor search radius (km)
    hidden_dims: list = [64, 64, 32],   # MLP hidden dimensions
    integrator_type: str = "symplectic_euler",  # or "stormer_verlet"
    dt: float = 60.0,                   # Time step (seconds)
    use_cutoff: bool = True,            # Use radius cutoff
    max_neighbors: int = 50,            # Max neighbors per node
    device: str = "cpu",                # torch device
    create_graph: bool = False,         # Enable backprop
    softplus_output: bool = True,       # Constrain Φ ≥ 0
    normalize_features: bool = False,   # Normalize MLP inputs
    init_gain: float = 0.1,             # Weight initialization scale
    use_dt_scaling: bool = False,       # dt²-structured correction
    alpha_init: float = 1e-4,           # Initial value for learnable α (used when use_dt_scaling=True)
)
```

### HGNN Methods

```python
# Forward propagation
q_final, p_final = model(q, p, n_steps=50)

# With trajectory
q_traj, p_traj = model(q, p, n_steps=50, return_trajectory=True)

# Compute Hamiltonian
H = model.compute_hamiltonian(q, p)

# Predict conjunction events
results = model.predict_conjunction(q, p, n_steps=100, threshold_distance=10.0)
```

### Ground Truth Hamiltonians

```python
from h_gnn.models.multibody import MultiBodyHamiltonian

gt = MultiBodyHamiltonian(
    mu=398600.4418,
    epsilon=100.0,      # Inter-satellite coupling strength
    satellite_mass=1000.0,
    softening=10.0,     # Prevent singularity (km)
)

H = gt(q, p, sum_over_particles=True)
grad_q, grad_p = gt.compute_gradients(q, p, edge_index)
```

---

## Configuration

### config.yaml

```yaml
# Physics parameters
mu: 398600.4418          # Earth's gravitational parameter
epsilon: 100.0           # Multi-body coupling strength

# Model architecture
hidden_dims: [64, 64, 32]
interaction_radius: 20000.0
interaction_strength: 0.01
use_dt_scaling: false

# Integration
integrator_type: stormer_verlet
dt: 60.0
n_steps: 50

# Training
n_epochs: 300
batch_size: 10
lr: 0.0001
horizons: [5, 10, 20, 50]
weights: [0.2, 0.3, 0.3, 0.2]
lambda_energy: 0.1

# Data
n_particles: 5
n_train: 1000
n_eval: 100
```

---

## Usage Guide

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/h-gnn.git
cd h-gnn

# Create environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: FAISS GPU
pip install faiss-gpu
```

### Quick Start

```python
import torch
from h_gnn.models.hgnn import HGNN

# Create model
model = HGNN(
    mu=398600.4418,
    interaction_radius=20000.0,
    use_dt_scaling=True,
)

# Generate initial conditions (5 satellites in LEO)
q = torch.randn(5, 3) * 7000  # ~7000 km altitude
p = torch.randn(5, 3) * 7.5   # ~7.5 km/s velocity

# Predict 50 steps ahead
q_pred, p_pred = model(q.float(), p.float(), n_steps=50)
```

### Training

```python
from h_gnn.experiments.multiscale_training import train_multiscale

config = {
    "n_particles": 5,
    "n_epochs": 300,
    "epsilon": 100.0,
}

model = train_multiscale(config)
torch.save(model.state_dict(), "hgnn_trained.pt")
```

### Evaluation

```python
from h_gnn.experiments.multiscale_training import evaluate_multiscale

results = evaluate_multiscale(model, config)

for T, metrics in results.items():
    print(f"T={T}: {metrics['improvement']:.1f}% improvement")
```

### Running Experiments

```bash
# ε-sweep scaling law
python -m h_gnn.experiments.epsilon_sweep

# Baseline comparison
python -m h_gnn.experiments.baseline_comparison

# Backward-error aligned training
python -m h_gnn.experiments.backward_error_training

# Validation tests
python -m h_gnn.experiments.validation_tests
```

---

## Theoretical Insights

### Why Symplectic Structure Matters

1. **Energy Conservation**: Symplectic integrators have bounded energy error
2. **Phase Space Preservation**: Volume-preserving (Liouville's theorem)
3. **Long-Horizon Stability**: No secular drift in invariants
4. **Learning Signal**: Allows network to focus on interaction learning

### Backward Error Interpretation

The learned H-GNN can be viewed as:
```
H_modified = H_true + O(dt²) corrections
```

Where the correction Φ_θ absorbs integration errors by learning the "modified Hamiltonian" that the integrator exactly preserves.

### Limitations

1. **Scale Sensitivity**: Performance depends on ε being in optimal range (25-50)
2. **Horizon Degradation**: Improvement decreases after T≈100 steps
3. **Seed Variance**: Results sensitive to initialization (23.9% std at T=100)
4. **N-Scaling**: Performance degrades for N≥25 particles

---

## Citation

If you use H-GNN in your research, please cite:

```bibtex
@article{hgnn2024,
  title={Hamiltonian Graph Neural Networks for Structure-Preserving Dynamics Learning},
  author={...},
  journal={...},
  year={2024}
}
```

---

## License

MIT License

---

## Contact

For questions or contributions, please open an issue on GitHub or contact the maintainers.
