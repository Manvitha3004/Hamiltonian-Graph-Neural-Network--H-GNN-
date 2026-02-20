# Hamiltonian Graph Neural Network (H-GNN)

A structure-preserving graph neural network for orbital mechanics and space situational awareness.

## Key Principles

1. **Hamiltonian must be scalar and invariant** - The learned Hamiltonian H outputs a single scalar and is SO(3) invariant
2. **Dynamics from ∇H only** - Time evolution comes exclusively from Hamilton's equations, NOT message passing
3. **Manual symplectic integration** - Custom symplectic integrators preserve phase space structure
4. **Independently ablatable components** - Every module can be toggled for controlled experiments
5. **Rare-regime evaluation prioritized** - Focus on generalization to unseen orbital regimes

## Architecture

Each node state: `z_i = (q_i, p_i)` where `q_i, p_i ∈ ℝ³`

Total Hamiltonian:
```
H_G = Σᵢ H_orbit(zᵢ) + λ Σ_{(i,j)∈E} Φ_θ(zᵢ, zⱼ)
```

Dynamics:
```
ż = J ∇H_G(z)
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/h-gnn.git
cd h-gnn

# Create environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Optional: Install FAISS for GPU
pip install faiss-gpu  # Requires CUDA

# Optional: Install e3nn for SO(3) equivariance (Phase 3)
pip install e3nn
```

## Quick Start

### Phase 1: Validate Symplectic Integration

```bash
python -m h_gnn.train --phase 1
```

Expected output:
- Symplectic Euler: bounded oscillatory energy error
- Vanilla Euler: linear energy drift (diverges)
- RK4: slow drift

### Phase 2: Train Graph Hamiltonian

```bash
python -m h_gnn.train --phase 2
```

### Phase 4: Rare-Regime Evaluation

```bash
python -m h_gnn.train --phase 4
```

### Full Pipeline

```bash
python -m h_gnn.train --config h_gnn/config.yaml
```

## Project Structure

```
h_gnn/
├── data/
│   ├── synthetic_catalog.py   # Orbital element generation
│   └── tle_parser.py          # TLE parsing and SGP4
├── models/
│   ├── hamiltonian.py         # Two-body Hamiltonian
│   ├── interaction.py         # Learned interaction potential
│   ├── graph.py               # FAISS graph construction
│   ├── hgnn.py                # Main H-GNN architecture
│   └── baselines.py           # GNN baselines for comparison
├── integrators/
│   ├── symplectic.py          # Symplectic Euler, Störmer-Verlet
│   └── baselines.py           # Vanilla Euler, RK4
├── experiments/
│   ├── rare_regime.py         # Rare-regime generalization
│   ├── scaling.py             # O(N log N) scaling analysis
│   └── equivariance_test.py   # SO(3) invariance validation
├── train.py                   # Main training script
├── eval.py                    # Evaluation metrics
└── config.yaml                # Configuration file
```

## Key Experiments

### 1. Integrator Comparison (Phase 1)

Validates that symplectic integrators preserve Hamiltonian structure:

```python
from h_gnn.integrators import SymplecticEuler, StormerVerlet
from h_gnn.integrators.baselines import VanillaEuler, RK4
from h_gnn.models import TwoBodyHamiltonian

hamiltonian = TwoBodyHamiltonian(mu=398600.4418)
integrators = {
    "Symplectic": SymplecticEuler(dt=60.0),
    "Verlet": StormerVerlet(dt=60.0),
    "Euler": VanillaEuler(dt=60.0),
    "RK4": RK4(dt=60.0),
}
```

### 2. Rare-Regime Generalization (Phase 4)

Tests whether Hamiltonian structure improves out-of-distribution performance:

- **Train**: Equatorial orbits (inclination 0-20°)
- **Test**: Polar orbits (inclination 60-90°)

Success criterion: H-GNN degrades ≥10% less than baseline GNNs.

### 3. Scaling Analysis

Target complexity: O(N log N) via FAISS radius search

```bash
python -m h_gnn.experiments.scaling --max-n 5000
```

## Configuration

Key parameters in `config.yaml`:

```yaml
physics:
  mu: 398600.4418      # Earth GM (km³/s²)

integration:
  dt: 60.0             # Time step (seconds)
  integrator: "symplectic_euler"

model:
  interaction:
    strength: 0.01     # λ coupling constant
    radius: 1000.0     # Interaction radius (km)
    hidden_dims: [64, 64, 32]
```

## Metrics (Non-Negotiable)

Every experiment must log:

- **Energy drift** per object
- **Angular momentum drift**
- **Rare-event recall** 
- **False negative rate**
- **Runtime scaling exponent**
- **Memory usage**

## Ablations

| Ablation | Command |
|----------|---------|
| Pure two-body (λ=0) | `model.disable_interaction()` |
| No symplectic | Use `VanillaEuler` integrator |
| Equivariant interaction | Enable `e3nn` in config |

## Expected Results

### Phase 1 Validation
- Symplectic: max energy drift < 1%
- Verlet: max energy drift < 0.1%
- Euler: diverges

### Phase 4 Evaluation
- If rare-regime improvement ≥ 10-15%: Strong ML conference submission
- If modest but stable: Structure-preserving physical learning paper
- If no improvement: Pivot to long-horizon simulation learning

## Citation

```bibtex
@article{hgnn2024,
  title={Hamiltonian Graph Neural Networks for Orbital Mechanics},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
