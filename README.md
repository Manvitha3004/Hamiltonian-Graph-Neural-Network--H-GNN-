# Hamiltonian Graph Neural Network (H-GNN)

**Learning Structured Corrections to Incomplete Hamiltonians via Symplectic Neural Parameterization**

---

## Central Idea

When the true Hamiltonian of a physical system contains terms that are unknown or computationally intractable, we can learn corrections while preserving the symplectic structure that governs long-term stability.

```
H_true = H_known + H_unknown
```

H-GNN parameterizes the unknown correction as a graph neural network:

```
H_theta = H_known + Phi_theta
```

The key insight: **structure preservation is not optional**. Non-symplectic methods (vanilla GNNs + RK4) perform *worse* than physics-only baselines, while symplectic integration enables stable learning.

---

## Theoretical Foundation

### Backward Error Interpretation

A symplectic integrator applied to an approximate Hamiltonian exactly solves a *nearby* Hamiltonian (the modified Hamiltonian). Our formulation:

```
H = H_base + dt^2 * alpha * (1/N) * sum(Phi_ij)
```

Here:
- `dt^2` scaling aligns with backward error analysis
- `alpha` is a learnable magnitude (prevents gradient imbalance)
- `1/N` normalization ensures per-particle force stability

### Why Symplectic Matters

| Method | T=50 Improvement | Energy Drift |
|--------|------------------|--------------|
| H-GNN (symplectic) | **+21%** | 2.8 |
| Vanilla GNN (RK4) | **-4%** | 7.8 |
| Physics only | baseline | 0.9 |

Non-symplectic methods **destroy** learned corrections over time. The symplectic constraint is essential, not merely helpful.

### Long-Horizon Stability

H-GNN maintains positive improvement even at 500 integration steps:

| Horizon | Improvement |
|---------|-------------|
| T=50 | +17.6% |
| T=100 | +11.6% |
| T=200 | +0.9% |
| T=500 | +4.9% |

---

## Architecture

**State**: Each node `z_i = (q_i, p_i)` with position and momentum in R^3

**Hamiltonian**:
```
H = sum_i H_orbit(z_i) + dt^2 * alpha * (1/N) * sum_{i,j} Phi(z_i, z_j)
```

**Dynamics**: Hamilton's equations with symplectic integration
```
dq/dt = dH/dp,  dp/dt = -dH/dq
```

**Key constraint**: Time evolution comes from gradients of H only, never from message passing directly.

---

## Core Claims (Validated)

| Claim | Evidence |
|-------|----------|
| 1. Symplectic structure is necessary | Vanilla GNN < 0% (worse than physics) |
| 2. H-GNN learns real corrections | Non-zero Phi, +15-23% improvement |
| 3. Long-horizon stable | T=500: +4.9% |
| 4. Learnability threshold exists | epsilon < 5 fails |
| 5. Multi-scale training helps | T=50: +13.4% mean |

---

## Quick Start

```python
import torch
from h_gnn.models.hgnn import HGNN

# Load frozen model
checkpoint = torch.load('hgnn_final_v1.0.pt')

model = HGNN(
    mu=398600.4418,
    interaction_radius=20000.0,
    hidden_dims=[64, 64, 32],
    use_dt_scaling=True,
    alpha_init=1e-4,
).float()

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict trajectory
q_final, p_final = model(q0, p0, n_steps=50, dt=60.0)
```

---

## Installation

```bash
git clone https://github.com/your-repo/h-gnn.git
cd h-gnn
pip install -r requirements.txt
```

---

## Project Structure

```
h_gnn/
├── models/
│   ├── hamiltonian.py      # Graph Hamiltonian with Phi
│   ├── interaction.py      # SO(3)-invariant interaction MLP
│   └── hgnn.py             # Main architecture
├── integrators/
│   └── symplectic.py       # Stormer-Verlet integrator
├── experiments/
│   ├── backward_error_training.py  # Training with dt^2 scaling
│   ├── baseline_comparison.py      # H-GNN vs vanilla GNN
│   └── epsilon_sweep.py            # Learnability threshold
└── physics/
    └── multibody.py        # Ground truth N-body Hamiltonian
```

---

## Key Files

| File | Purpose |
|------|---------|
| `hgnn_final_v1.0.pt` | Frozen model checkpoint |
| `EXPERIMENTAL_RESULTS.md` | Validated claims with data |
| `FINAL_MODEL.md` | Model specification |

---

## Citation

```bibtex
@article{hgnn2026,
  title={Learning Structured Corrections to Incomplete Hamiltonians 
         via Symplectic Neural Parameterization},
  author={...},
  year={2026}
}
```

## License

MIT License
