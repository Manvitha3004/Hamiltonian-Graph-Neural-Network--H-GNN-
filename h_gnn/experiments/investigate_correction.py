"""Quick investigation of correction magnitudes."""

import torch
import numpy as np
from pathlib import Path

from h_gnn.models import HGNN, TwoBodyHamiltonian
from h_gnn.models.multibody import MultiBodyHamiltonian

print('CORRECTION MAGNITUDE INVESTIGATION')
print('='*60)

# Load model
hgnn = HGNN(
    mu=398600.4418,
    interaction_strength=1.0,
    interaction_radius=5000.0,
    hidden_dims=[128, 128, 64],
    integrator_type='stormer_verlet',
    dt=60.0,
    softplus_output=False,
    normalize_features=True,
).double()

model_path = Path('hgnn_multibody.pt')
hgnn.load_state_dict(torch.load(model_path, weights_only=True))
print(f'Loaded model from {model_path}')

# Check lambda value
lam = hgnn.hamiltonian.lambda_.item()
print(f'lambda (interaction strength): {lam:.4f}')

# Create sample state
n_particles = 5
np.random.seed(42)
r = 7000 + 500 * np.random.randn(n_particles)
theta = np.linspace(0, 2*np.pi, n_particles, endpoint=False)

q = torch.tensor([
    [r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i]), 50.0 * np.random.randn()]
    for i in range(n_particles)
], dtype=torch.float64)

v_circ = np.sqrt(398600.4418 / r)
p = torch.tensor([
    [v_circ[i] * (-np.sin(theta[i])), v_circ[i] * np.cos(theta[i]), 0.1 * np.random.randn()]
    for i in range(n_particles)
], dtype=torch.float64)

# Build edge index
edges = [[i, j] for i in range(n_particles) for j in range(n_particles) if i != j]
edge_index = torch.tensor(edges, dtype=torch.long).T

# Get gradients
with torch.no_grad():
    tb = TwoBodyHamiltonian()
    mb = MultiBodyHamiltonian(epsilon=50.0)
    
    grad_q_tb, _ = tb.compute_gradients(q, p)
    grad_q_mb, _ = mb.compute_gradients(q, p, edge_index)
    grad_q_hgnn, _ = hgnn.hamiltonian.compute_gradients(q, p, edge_index)

print()
print('GRADIENT MAGNITUDES')
print('-'*40)
print(f'||grad_H_twobody||: {torch.norm(grad_q_tb).item():.6f}')
print(f'||grad_H_multibody||: {torch.norm(grad_q_mb).item():.6f}')
print(f'||grad_H_learned||: {torch.norm(grad_q_hgnn).item():.6f}')

print()
print('CORRECTION MAGNITUDES')
print('-'*40)
phi_true = grad_q_mb - grad_q_tb
phi_learned = grad_q_hgnn - grad_q_tb
print(f'||grad_Phi_true||:    {torch.norm(phi_true).item():.6f}')
print(f'||grad_Phi_learned||: {torch.norm(phi_learned).item():.6f}')
print(f'Ratio (learned/true): {torch.norm(phi_learned).item() / (torch.norm(phi_true).item() + 1e-10):.4f}')

print()
print('DIRECT Phi OUTPUT TEST')
print('-'*40)
# Test Phi directly
q_i = q[edge_index[0]]
p_i = p[edge_index[0]]
q_j = q[edge_index[1]]
p_j = p[edge_index[1]]

phi_out = hgnn.hamiltonian.Phi(q_i.float(), p_i.float(), q_j.float(), p_j.float())
print(f'Phi network output: min={phi_out.min().item():.6f}, max={phi_out.max().item():.6f}')
print(f'Phi network mean: {phi_out.mean().item():.6f}')
print(f'Phi network std: {phi_out.std().item():.6f}')

print()
print('MODEL PARAMETER STATISTICS')
print('-'*40)
total_params = sum(p.numel() for p in hgnn.parameters())
trainable = sum(p.numel() for p in hgnn.parameters() if p.requires_grad)
print(f'Total parameters: {total_params}')
print(f'Trainable parameters: {trainable}')

# Check MLP weights
for name, param in hgnn.hamiltonian.Phi.named_parameters():
    if 'weight' in name:
        print(f'{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}')
