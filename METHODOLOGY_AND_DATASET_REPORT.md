# Methodology and Dataset Report

Date: March 16, 2026
Project: Hamiltonian Graph Neural Network (H-GNN)

---

## 1. Objective

This report documents:

1. What datasets were used
2. What each dataset contains
3. How data is processed into training samples
4. The full methodology used for model training and evaluation

---

## 2. Datasets Used

The project uses two dataset families:

1. Real-world TLE/SGP4 trajectories (`satellites.tle`)
2. Synthetic orbital datasets (controlled experiments)

---

## 3. Real Dataset: TLE + SGP4

Primary implementation: `h_gnn/data/tle_loader.py`

### 3.1 Source

1. Input file format: Three-line TLE blocks (name + line1 + line2)
2. Propagation engine: SGP4 (`sgp4.api.Satrec`)
3. Coordinate/state output: position (km) and velocity (km/s)

### 3.2 Satellite selection

From parsed TLE entries, satellites are filtered to LEO for most runs with:

1. Mean motion > 11 rev/day
2. Eccentricity < 0.1

Then a subset of `n_satellites` is sampled.

### 3.3 Temporal sampling

Typical defaults in training/experiments:

1. `step_minutes = 1`
2. `duration_minutes` varies by experiment (for example 60, 120, 480, 720, 1440)

Each satellite is propagated over the same time window; all trajectories are truncated to the shortest valid propagated length across selected satellites.

### 3.4 Raw tensor structure

After propagation, arrays are assembled as:

1. `positions`: shape `(n_satellites, T, 3)` in km
2. `velocities`: shape `(n_satellites, T, 3)` in km/s

Metadata stored:

1. satellite names
2. number of satellites
3. number of timesteps
4. start time
5. step and duration settings

### 3.5 Supervised sample construction

Function: `convert_to_hgnn_format(...)`

Given `horizon` and `stride`, sliding windows are used:

1. Input state at time `t`: `(q0, p0)`
2. Target state at time `t + horizon`: `(q_target, p_target)`

Returned tensors:

1. `q0`: `(n_samples, n_particles, 3)`
2. `p0`: `(n_samples, n_particles, 3)`
3. `q_target`: `(n_samples, n_particles, 3)`
4. `p_target`: `(n_samples, n_particles, 3)`

### 3.6 Dataset windowing formula

Let trajectories be sampled at uniform timestep `delta_t` (1 minute), with:

1. `Q in R^(N x T x 3)` for positions
2. `P in R^(N x T x 3)` for velocities

For a prediction horizon `H` (in steps) and stride `S`, each sample index `k` is generated from start time `t_k = kS`:

1. `q0^(k) = Q[:, t_k, :]`
2. `p0^(k) = P[:, t_k, :]`
3. `q_target^(k) = Q[:, t_k + H, :]`
4. `p_target^(k) = P[:, t_k + H, :]`

with valid starts:

`t_k in {0, S, 2S, ..., T - H - 1}`

Hence the number of samples is:

`n_samples = floor((T - H - 1)/S) + 1`

---

## 4. Synthetic Datasets (Controlled Studies)

Primary implementations:

1. `h_gnn/data/synthetic_catalog.py`
2. `h_gnn/models/multibody.py`

### 4.1 SyntheticCatalog dataset

`SyntheticCatalog` generates orbital elements and converts to Cartesian states.

Each object is sampled by Keplerian elements:

1. Semi-major axis from altitude range
2. Eccentricity (low-to-moderate)
3. Inclination range (configurable)
4. RAAN, argument of periapsis, true anomaly

Converted outputs:

1. `q` (position, km)
2. `p` (velocity, km/s)

### 4.2 Regime-based synthetic splits

Used for generalization and rare-regime tests:

1. Training-like inclination regimes (for example equatorial/low inclination)
2. OOD test regimes (high inclination/polar)

### 4.3 Multi-body synthetic ground truth

`MultiBodyHamiltonian` provides controlled inter-satellite interaction signal:

1. Baseline two-body term
2. Pairwise interaction potential with strength `epsilon`
3. Softening term to avoid singularities

This is used for:

1. epsilon sweep studies
2. scaling-law studies
3. controlled validation of correction learnability

---

## 5. Dataset Composition in Practice

### 5.1 Real TLE runs (observed examples)

In recorded project runs, the TLE pipeline reported:

1. Parsed satellites: 14,660
2. LEO-filtered satellites: 13,854

Example small-run composition from logs:

1. `n_satellites = 3`
2. `duration_minutes = 60`
3. `horizon = 15`
4. Generated supervised samples: 15
5. Split: 12 train / 3 validation

### 5.2 Sample semantics

Each supervised sample consists of all chosen satellites together (graph snapshot):

1. Node state per satellite: 3D position + 3D velocity
2. Target: future node states after horizon steps

So each sample is a multi-body state transition problem, not independent single-satellite forecasting.

---

## 6. Preprocessing and Normalization

### 6.1 Physical-unit training

Models operate in physical units (km and km/s), with normalization applied in loss/features for stability.

Normalized variables used internally in loss and feature scaling:

1. `q_tilde = q / Q_SCALE`
2. `p_tilde = p / P_SCALE`

where typically:

1. `Q_SCALE = 10000.0 km`
2. `P_SCALE = 10.0 km/s`

### 6.2 Typical scaling constants

From `train_on_tle_improved.py` and experiment code:

1. `Q_SCALE = 10000.0` km
2. `P_SCALE = 10.0` km/s

Used for:

1. trajectory-loss normalization
2. noise scaling in robustness tests
3. stable MLP feature magnitudes

### 6.3 Splitting

Typical split used in training scripts:

1. 80% training
2. 20% validation

Split strategies used across scripts:

1. Random permutation split (`train_on_tle_improved.py`)
2. Prefix/suffix split in some experiment functions (`run_experiments.py`)

---

## 7. Methodology: Modeling Approach

## 7.1 Core model

H-GNN models a corrected Hamiltonian:

`H_theta = H_base + dt^2 * alpha * (1/N) * sum Phi_ij`

More explicitly for `N` satellites:

`H_theta(q, p) = H_base(q, p) + dt^2 * alpha * (1/N) * sum_(i != j) Phi_theta(z_i, z_j)`

with state `z_i = (q_i, p_i)`.

The base two-body Hamiltonian is:

`H_base(q, p) = sum_(i=1..N) [ 0.5 * ||p_i||^2 - mu / ||q_i|| ]`

Hamilton's equations define dynamics:

1. `dq_i/dt = partial H_theta / partial p_i`
2. `dp_i/dt = - partial H_theta / partial q_i`

Where:

1. `H_base`: two-body Hamiltonian
2. `Phi_ij`: learnable pairwise interaction potential on graph edges
3. `alpha`: learnable correction magnitude parameter
4. `dt^2` scaling aligns with symplectic integrator backward-error behavior

### 7.2 Integrator

Dynamics rollout uses symplectic integration (Stormer-Verlet family in `h_gnn/integrators/symplectic.py`) to preserve phase-space structure and reduce long-horizon drift.

For one step from `(q_n, p_n)` to `(q_(n+1), p_(n+1))` with step `dt`, Stormer-Verlet is:

1. `p_(n+1/2) = p_n - (dt/2) * grad_q H_theta(q_n, p_n)`
2. `q_(n+1)   = q_n + dt * grad_p H_theta(q_n, p_(n+1/2))`
3. `p_(n+1)   = p_(n+1/2) - (dt/2) * grad_q H_theta(q_(n+1), p_(n+1/2))`

This is a second-order symplectic scheme used for stable long-horizon rollout.

### 7.3 Graph representation

At each state, satellites are treated as nodes, and pairwise interactions are learned through graph edges and invariant feature mappings.

---

## 8. Methodology: Training Protocol

Primary stable protocol (real TLE training): `train_on_tle_improved.py`

### 8.1 Key hyperparameters (typical)

1. Optimizer: AdamW
2. Learning rate: `1e-4`
3. Gradient clipping: max norm `1.0`
4. Timestep: `dt = 60.0` seconds (1 minute)
5. Hidden dimensions: `[128, 128, 64]`
6. `alpha_init = 0.5` (raw, then constrained internally)

### 8.2 Loss function

Total loss combines:

1. Normalized trajectory loss:
   1. position error scaled by `Q_SCALE`
   2. velocity error scaled by `P_SCALE`
2. Energy conservation regularization (small weight, typically `1e-6`)

Per-sample normalized trajectory loss:

1. `L_pos = mean_i ||q_pred,i - q_target,i|| / Q_SCALE`
2. `L_vel = mean_i ||p_pred,i - p_target,i|| / P_SCALE`
3. `L_traj = L_pos + 0.1 * L_vel`

Specific orbital energy per particle:

`E_i(q_i, p_i) = 0.5 * ||p_i||^2 - mu / ||q_i||`

Energy regularization (normalized drift):

1. `E0 = E(q0, p0)`
2. `Epred = E(q_pred, p_pred)`
3. `E_scale = mean(|E0|) + eps`
4. `L_energy = mean( ((Epred/E_scale) - (E0/E_scale))^2 )`

Total training objective:

`L_total = L_traj + lambda_E * L_energy`

where typically `lambda_E = 1e-6`.

Optimization update:

1. `theta <- AdamW(theta, grad_theta L_total)`
2. Gradient clipping: `||grad||_2 <= 1.0`

### 8.3 Validation and model selection

1. Per-epoch validation loss is tracked
2. Best checkpoint saved as `best_hgnn_tle.pt`
3. Final model saved as `trained_hgnn_tle_improved.pt`

---

## 9. Methodology: Experimental Evaluation Design

Main runner: `experiments/run_experiments.py`

### 9.1 Baselines

1. Linear extrapolation baseline (`q + v*dt`)
2. Physics-only two-body baseline (velocity Verlet)
3. Pure neural network baseline (MLP, no Hamiltonian structure)
4. H-GNN

Baseline formulas:

1. Linear extrapolation:
   `q_hat = q0 + p0 * (H * dt), p_hat = p0`
2. Physics-only two-body:
   integrates `H_base` with velocity-Verlet over `H` steps
3. Pure NN baseline:
   `f_theta(q0, p0) -> (Delta q, Delta p)`, then
   `q_hat = q0 + Delta q, p_hat = p0 + Delta p`

### 9.2 Experiment types

1. Long-horizon prediction (vary horizon)
2. Scaling experiment (vary number of satellites)
3. Noise robustness (perturb initial conditions)
4. Visualization and comparison-table generation

### 9.3 Metrics

Computed metrics include:

1. RMSE/MAE/Max position error (km)
2. RMSE/MAE/Max velocity error (km/s)
3. Energy drift
4. Percent improvement vs baseline

Metric equations for a validation set of `M` samples:

1. Position error per sample:
   `e_pos^(m) = mean_i ||q_hat_i^(m) - q_i^(m)||`
2. Velocity error per sample:
   `e_vel^(m) = mean_i ||p_hat_i^(m) - p_i^(m)||`
3. RMSE position:
   `RMSE_pos = sqrt( (1/M) * sum_m (e_pos^(m))^2 )`
4. RMSE velocity:
   `RMSE_vel = sqrt( (1/M) * sum_m (e_vel^(m))^2 )`
5. MAE position:
   `MAE_pos = (1/M) * sum_m e_pos^(m)`
6. Relative energy drift:
   `Drift_E = mean_m |E_hat^(m) - E0^(m)| / (|E0^(m)| + eps)`
7. Improvement (%) vs baseline `B`:
   `Improvement = 100 * (Error_B - Error_HGNN) / Error_B`

Positive improvement means H-GNN is better; negative means worse.

---

## 10. Dataset and Methodology Summary (Concise)

1. Real data: TLE satellites propagated with SGP4 into multivariate time series of positions and velocities.
2. Sample format: sliding-window state-to-future-state graph samples for all selected satellites together.
3. Synthetic data: controlled orbital catalogs and multi-body interaction simulators for theory-driven experiments.
4. Model method: learn Hamiltonian correction on graph interactions, integrate with symplectic solver.
5. Training method: normalized trajectory loss + light energy regularization + gradient clipping + validation checkpointing.
6. Evaluation method: compare against linear, physics-only, and pure neural baselines across horizons, scale, and perturbation settings.

---

## 11. Compact Formula Sheet

This section collects the key formulas in one place.

1. Base Hamiltonian:
   `H_base = sum_i (0.5 ||p_i||^2 - mu/||q_i||)`
2. Corrected Hamiltonian:
   `H_theta = H_base + dt^2 * alpha * (1/N) * sum_(i!=j) Phi_theta(z_i, z_j)`
3. Dynamics:
   `dq/dt = partial H_theta / partial p`, `dp/dt = - partial H_theta / partial q`
4. Stormer-Verlet:
   `p_half = p - 0.5 dt grad_q H`,
   `q_new = q + dt grad_p H(q, p_half)`,
   `p_new = p_half - 0.5 dt grad_q H(q_new, p_half)`
5. Training loss:
   `L_total = (L_pos + 0.1 L_vel) + lambda_E L_energy`
6. Improvement:
   `100 * (Error_baseline - Error_HGNN) / Error_baseline`

---

## 12. Reproducibility References

Key files to reproduce dataset and methodology exactly:

1. `h_gnn/data/tle_loader.py`
2. `h_gnn/data/synthetic_catalog.py`
3. `h_gnn/models/multibody.py`
4. `h_gnn/models/hgnn.py`
5. `train_on_tle_improved.py`
6. `experiments/run_experiments.py`
7. `results/` JSON artifacts for all completed experiments
