# Hamiltonian Graph Neural Network (H-GNN)

**Learning Structured Corrections to Incomplete Hamiltonians via Symplectic Neural Parameterization**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Key Result

**H-GNN achieves +17-23% trajectory prediction improvement** over physics-only baselines, while vanilla neural networks (without symplectic constraints) perform **worse than baselines (-4%)**.

| Method | T=50 Improvement | Energy Drift |
|--------|------------------|--------------|
| **H-GNN (symplectic)** | **+21.3%** | 2.97 |
| Vanilla GNN (RK4) | -4.2% | 7.81 |
| Physics only | baseline | 0.90 |

---

## 💡 Central Idea

When the true Hamiltonian of a physical system contains terms that are unknown or computationally intractable, we can learn corrections while preserving the symplectic structure that governs long-term stability.

```
H_true = H_known + H_unknown
```

H-GNN parameterizes the unknown correction as a graph neural network:

```
H_θ = H_base + dt² × α × (1/N) × Σ Φ(zᵢ, zⱼ)
```

**The key insight**: Structure preservation is **not optional**. Non-symplectic methods destroy learned corrections over time. The symplectic constraint is essential, not merely helpful.

---

## 🔬 Theoretical Foundations

### 1. Learnability Threshold Theorem

Learning succeeds only when signal-to-noise ratio exceeds unity:

```
SNR = (ε × T × ||∇Φ||) / σ_noise > 1
```

**Empirical validation**: Below ε ≈ 5, learning fails consistently.

| ε | Improvement | Status |
|---|-------------|--------|
| 1 | -1.9% | ❌ FAIL |
| 5 | +15.0% | ✅ PASS |
| 25 | +23.4% | ⭐ OPTIMAL |

### 2. Non-Symplectic Drift Theorem

For non-symplectic integrators, energy drift grows linearly:
```
ΔE_non-symplectic(T) ≥ c × T × dt^(k-1)
```

For symplectic integrators, energy remains bounded:
```
ΔE_symplectic(T) ≤ C × dt^k  (independent of T)
```

### 3. Backward Error Alignment

The dt² scaling matches Störmer-Verlet's modified Hamiltonian structure:
```
H_modified = H + dt² × H₂ + dt⁴ × H₄ + ...
```

This prevents gradient scale mismatch and enables stable training.

---

## 🏗️ Architecture

### State Representation
Each particle i has state `zᵢ = (qᵢ, pᵢ)`:
- `qᵢ ∈ ℝ³`: position (km)
- `pᵢ ∈ ℝ³`: momentum/velocity (km/s)

### Hamiltonian Structure
```
H = Σᵢ H_orbit(zᵢ) + dt² × α × (1/N) × Σᵢⱼ Φ(zᵢ, zⱼ)
```

### Interaction Potential Φ
Uses SO(3)-invariant features:
- `r = ||qⱼ - qᵢ||` (pairwise distance)
- `r²` (squared distance)
- `||Δp||` (momentum difference norm)
- `Δq · Δp` (position-momentum correlation)
- `Δq · p_avg` (average momentum projection)

**MLP**: Input(5) → 64 → 64 → 32 → 1 with SiLU activations

### Integrator: Störmer-Verlet
Second-order symplectic (leapfrog):
```
p_half = pₙ - (dt/2) × ∇_q H(qₙ, pₙ)
q_{n+1} = qₙ + dt × ∇_p H(qₙ, p_half)
p_{n+1} = p_half - (dt/2) × ∇_q H(q_{n+1}, p_half)
```

---

## 📊 Experimental Results

### Long-Horizon Stability
H-GNN maintains positive improvement even at 500 integration steps:

| Horizon | Improvement |
|---------|-------------|
| T=50 | +17.6% |
| T=100 | +11.6% |
| T=200 | +0.9% |
| T=500 | +4.9% |

### Energy Conservation

| Horizon | H-GNN Drift | Vanilla GNN Drift | Ratio |
|---------|-------------|-------------------|-------|
| T=10 | 0.30 | 2.83 | 9.4× |
| T=25 | 1.00 | 7.64 | 7.6× |
| T=50 | 2.97 | 7.81 | 2.6× |

### Validated Claims

| Claim | Evidence |
|-------|----------|
| ✅ Symplectic structure is necessary | Vanilla GNN < 0% (worse than physics) |
| ✅ H-GNN learns real corrections | Non-zero Φ, +15-23% improvement |
| ✅ Long-horizon stable | T=500: +4.9% |
| ✅ Learnability threshold exists | ε < 5 fails |
| ✅ Multi-scale training helps | Mean improvement +13.4% |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Manvitha3004/Hamiltonian-Graph-Neural-Network--H-GNN-.git
cd Hamiltonian-Graph-Neural-Network--H-GNN-
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from h_gnn.models.hgnn import HGNN

# Create model
model = HGNN(
    mu=398600.4418,              # Earth's gravitational parameter (km³/s²)
    hidden_dims=[64, 64, 32],    # MLP architecture
    dt=60.0,                     # 60 second timestep
    use_dt_scaling=True,         # Backward-error aligned
    create_graph=True,           # Enable gradient computation
).float()

# Initial conditions: 10 satellites
q0 = torch.randn(10, 3) * 10000 + 10000  # positions (km)
p0 = torch.randn(10, 3) * 5               # velocities (km/s)

# Predict 50 steps forward
q_final, p_final = model(q0, p0, n_steps=50)
```

### Load Trained Model

```python
# Load frozen model checkpoint
checkpoint = torch.load('hgnn_final_v1.0.pt')

model = HGNN(
    mu=398600.4418,
    hidden_dims=[64, 64, 32],
    use_dt_scaling=True,
).float()

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    q_pred, p_pred = model(q0, p0, n_steps=100)
```

---

## 📁 Project Structure

```
h_gnn/
├── models/
│   ├── hgnn.py             # Main H-GNN architecture
│   ├── hamiltonian.py      # TwoBody and Graph Hamiltonians
│   ├── interaction.py      # Learned interaction potentials (MLP)
│   └── multibody.py        # Ground truth N-body Hamiltonian
├── integrators/
│   └── symplectic.py       # Störmer-Verlet and Symplectic Euler
├── experiments/
│   ├── epsilon_sweep.py            # Learnability threshold experiment
│   ├── baseline_comparison.py      # H-GNN vs Vanilla GNN
│   ├── generate_figures.py         # Publication-quality plots
│   └── theoretical_verification.py # Theorem validation
├── data/
│   └── trajectories.py     # Data generation utilities
└── tests/
    └── ...                 # Unit tests

figures/                    # Generated plots (PDF/PNG)
results/                    # Experiment outputs (JSON)
```

---

## 📈 Generated Figures

| Figure | Description |
|--------|-------------|
| `fig1_improvement_vs_horizon.pdf` | Main result: H-GNN vs Vanilla vs Physics |
| `fig2_epsilon_sweep.pdf` | Learnability threshold demonstration |
| `fig3_energy_drift.pdf` | Energy conservation comparison |
| `fig4_volume_preservation.pdf` | Symplectic structure verification |
| `fig5_improvement_vs_n.pdf` | Scaling with number of particles |
| `fig6_long_horizon.pdf` | Stability up to T=500 |
| `theorem1_snr_threshold.pdf` | SNR threshold visualization |
| `theorem2_symplectic_bound.pdf` | Energy drift bounds |
| `theorem3_dt_squared.pdf` | Backward error alignment |

---

## 📚 Key Documentation

| File | Description |
|------|-------------|
| `THEORETICAL_FOUNDATIONS.md` | Mathematical proofs and theorems |
| `EXPERIMENTAL_RESULTS.md` | Validated claims with data |
| `FINAL_MODEL.md` | Model specification and usage |
| `SCALING_DISCOVERY.md` | Scaling law experiments |
| `ISSC_PROJECT_DOCUMENTATION.txt` | Complete project summary |

---

## 🔮 Future Directions

1. **Universal SNR Threshold**: Test if SNR_crit is constant across all configurations
2. **Larger Systems**: Scale to 100+ particles with sparse graphs
3. **Real Data**: Apply to actual satellite TLE catalog data
4. **Adaptive Timestep**: Integration with adaptive symplectic methods
5. **Higher-Order Integrators**: 4th-order (Forest-Ruth, Yoshida) with dt⁴ parameterization

---

## 📖 References

1. Greydanus et al. "Hamiltonian Neural Networks" (NeurIPS 2019)
2. Sanchez-Gonzalez et al. "Hamiltonian Graph Networks" (NeurIPS 2019 Workshop)
3. Chen et al. "Symplectic Recurrent Neural Networks" (ICLR 2020)
4. Hairer et al. "Geometric Numerical Integration" (Springer, 2006)
5. Toth et al. "Hamiltonian Generative Networks" (ICLR 2020)

---

## 📄 Citation

```bibtex
@software{hgnn2026,
  title={H-GNN: Learning Structured Corrections to Incomplete Hamiltonians 
         via Symplectic Neural Parameterization},
  author={Manvitha},
  year={2026},
  url={https://github.com/Manvitha3004/Hamiltonian-Graph-Neural-Network--H-GNN-}
}
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

This project demonstrates that the failure of vanilla neural networks on Hamiltonian learning is **not** due to insufficient capacity, but due to **fundamental incompatibility** with Hamiltonian structure. Symplectic constraints convert an impossible problem into a tractable one.
