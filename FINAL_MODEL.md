# H-GNN Final (v1.0)

**Symplectic Neural Parameterization of Hamiltonian Corrections**

---

## Core Formulation

The model learns structured corrections to an incomplete Hamiltonian:

```
H_theta = H_known + dt^2 * alpha * (1/N) * sum_{i,j} Phi(z_i, z_j)
```

| Term | Interpretation |
|------|----------------|
| `H_known` | Two-body Kepler Hamiltonian (known physics) |
| `Phi(z_i, z_j)` | Learned pairwise interaction (unknown physics) |
| `dt^2` | Backward error scaling (matches integrator order) |
| `alpha` | Learnable magnitude (prevents gradient imbalance) |
| `1/N` | Per-particle normalization (force stability) |

**Key insight**: Symplectic integration applied to H_theta exactly solves a nearby modified Hamiltonian, ensuring long-horizon stability.

---

## Model Checkpoint

**File**: `hgnn_final_v1.0.pt`

## Architecture

| Component | Value |
|-----------|-------|
| Hidden dims | [64, 64, 32] |
| dt^2 scaling | Enabled |
| Learnable alpha | nn.Parameter |
| alpha init | 1e-4 |
| 1/N normalization | per-particle |
| Integrator | Stormer-Verlet (2nd order symplectic) |
| Parameters | 6,659 |

## Training Configuration

| Setting | Value |
|---------|-------|
| Method | Backward-error aligned |
| Loss | Trajectory MSE + energy drift |
| Horizons | [5, 10, 20, 50] |
| Weights | [0.2, 0.3, 0.3, 0.2] |
| Learning rate | 5e-5 |
| alpha regularization | 1e-4 * alpha^2 |
| Epochs | 200 |

## Usage

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

# Predict corrected trajectory
q_final, p_final = model(q0, p0, n_steps=50, dt=60.0)
```

## Status

| Property | Value |
|----------|-------|
| Version | 1.0.0 |
| Frozen | Yes |
| Date | 2026-02-20 |

---

## Why This Works

1. **Structure preservation**: Symplectic integration conserves phase space volume
2. **Backward error**: The integrator exactly solves H_modified = H_theta + O(dt^2)
3. **Learnable magnitude**: alpha adapts to the true correction scale
4. **Multi-scale training**: Horizons [5, 10, 20, 50] prevent short-horizon overfitting
