# Scaling Law Discovery Framework

**Goal:** Discover fundamental scaling laws and structural limitations in Hamiltonian correction learning.

This is NOT about improving model performance. This is about scientific discovery.

## Quick Start

```bash
# Run all experiments (full analysis, ~2-4 hours)
python h_gnn/experiments/run_discovery.py --all

# Quick test run (~20-30 minutes)
python h_gnn/experiments/run_discovery.py --quick

# Run specific experiments
python h_gnn/experiments/run_discovery.py --epsilon         # Epsilon threshold
python h_gnn/experiments/run_discovery.py --n-scaling       # System size scaling
python h_gnn/experiments/run_discovery.py --lyapunov        # Chaos analysis
python h_gnn/experiments/run_discovery.py --dt              # Backward error validation
python h_gnn/experiments/run_discovery.py --failure         # Failure modes
python h_gnn/experiments/run_discovery.py --cross           # Cross-parameter sweep

# Generate plots from existing results
python h_gnn/experiments/run_discovery.py --plot
```

## Experiments

### 1. Learnability Threshold Analysis (`--epsilon`)

**Question:** At what interaction strength ε does correction learning transition from failure to success?

**Method:**
- Sweep ε over logarithmic grid (0.1 → 300)
- Train with 3 seeds per ε
- Measure: T=50 improvement, gradient SNR, α convergence

**Output:**
- Critical threshold ε_crit
- Scaling exponent: improvement ~ ε^γ
- `results/scaling_laws/epsilon_threshold.json`

### 2. System Size Scaling (`--n-scaling`)

**Question:** How does correction learning scale with particle count N?

**Method:**
- Test N ∈ {10, 20, 40, 80, 160}
- Track: training stability, gradient magnitudes, α convergence

**Output:**
- N_crit where learning fails
- α scaling: α ~ N^γ
- `results/scaling_laws/n_scaling.json`

### 3. Chaos and Lyapunov Analysis (`--lyapunov`)

**Question:** Does symplectic correction reduce chaotic divergence?

**Method:**
- Estimate largest Lyapunov exponent via perturbation growth
- Compare: ground truth, physics-only, H-GNN

**Output:**
- Lyapunov exponents for each model
- `results/scaling_laws/lyapunov_analysis.json`

### 4. Backward Error dt Validation (`--dt`)

**Question:** Does the learned correction scale as dt²?

**Method:**
- Train at multiple dt values (15, 30, 60, 120)
- Fit α scaling with dt

**Output:**
- α ~ dt^γ (expected γ ≈ 2)
- `results/scaling_laws/dt_scaling.json`

### 5. Failure Mode Characterization (`--failure`)

**Question:** What are the precise conditions for learning failure?

**Method:**
- Systematically probe parameter boundaries
- Test: Φ collapse, α divergence, N-scaling failure, ε failure

**Output:**
- Failure boundaries
- `results/scaling_laws/failure_modes.json`

### 6. Cross-Parameter Sweep (`--cross`)

**Question:** What is the combined scaling law ε_crit(N, T, dt)?

**Method:**
- Full factorial sweep over (N, T, dt, ε)
- Fit multivariate regression

**Output:**
- Combined scaling law
- Invariant dimensionless groups
- `results/scaling_laws/cross_parameter_sweep.json`

## Metrics Tracked

For each training run, the framework logs:

| Metric | Description |
|--------|-------------|
| `final_improvement` | % improvement over physics-only baseline at T=50 |
| `final_alpha` | Learned correction magnitude |
| `grad_phi_norms` | ||∇Φ|| per epoch |
| `grad_hbase_norms` | ||∇H_base|| proxy per epoch |
| `gradient_snr` | Signal-to-noise ratio ||∇Φ|| / ||∇H_base|| |
| `loss_history` | Training loss per epoch |
| `alpha_history` | α value per epoch |
| `convergence_epoch` | Epoch where loss stabilized |
| `phi_collapsed` | Whether Φ output went to zero |
| `alpha_diverged` | Whether α exploded |

## Expected Outputs

After running `--all`, you should have:

```
results/scaling_laws/
├── epsilon_threshold.json     # Learnability threshold data
├── n_scaling.json             # System size scaling data
├── lyapunov_analysis.json     # Chaos analysis data
├── dt_scaling.json            # Backward error validation
├── failure_modes.json         # Failure mode boundaries
├── cross_parameter_sweep.json # Full parameter sweep
├── master_results_*.json      # Combined results
├── SCALING_LAWS_REPORT.md     # Summary report
└── figures/
    ├── epsilon_threshold.png/pdf
    ├── n_scaling.png/pdf
    ├── lyapunov_comparison.png/pdf
    ├── dt_scaling.png/pdf
    └── failure_modes.png/pdf
```

## Theoretical Predictions

The framework tests these theoretical predictions:

### Theorem 1: Learnability Threshold
The signal-to-noise ratio must exceed 1 for learning:
$$\text{SNR} = \frac{\varepsilon \cdot T \cdot \|\nabla\Phi\|}{\sigma_{\text{noise}}} > 1$$

### Theorem 2: Non-Symplectic Drift
Without symplectic structure, energy drift grows at least linearly:
$$\Delta E(T) \geq c \cdot T$$

### Theorem 3: Backward Error Alignment
The learned correction should scale as dt²:
$$\alpha \sim \mathcal{O}(dt^2)$$

## Key Files

| File | Purpose |
|------|---------|
| `scaling_law_discovery.py` | Core experiments (1-5) |
| `cross_parameter_analysis.py` | Cross-parameter sweep |
| `scaling_visualization.py` | Plot generation |
| `run_discovery.py` | Unified CLI runner |

## Example Analysis

```python
import json
from pathlib import Path

# Load results
with open("results/scaling_laws/epsilon_threshold.json") as f:
    data = json.load(f)

# Get critical threshold
eps_crit = data['scaling_law']['critical_value']
print(f"Critical epsilon: {eps_crit:.2f}")

# Get scaling exponent
gamma = data['scaling_law']['scaling_exponent']
print(f"Improvement ~ epsilon^{gamma:.3f}")
```
