"""
Synthetic Orbital Data Generation.

Generates orbital states from Keplerian elements with controlled
distributions for rare-regime evaluation.

Key features:
- Train on equatorial orbits (inclination 0-20°)
- Test on polar orbits (inclination 60-90°)
- Support for conjunction scenarios
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Union
import warnings


@dataclass
class OrbitalElements:
    """
    Classical Keplerian orbital elements.
    
    Attributes:
        a: Semi-major axis (km)
        e: Eccentricity (0 to <1 for elliptical)
        i: Inclination (radians)
        Omega: Right ascension of ascending node (radians)
        omega: Argument of periapsis (radians)
        nu: True anomaly (radians)
    """
    a: float          # Semi-major axis [km]
    e: float          # Eccentricity [0, 1)
    i: float          # Inclination [rad]
    Omega: float      # RAAN [rad]
    omega: float      # Argument of periapsis [rad]
    nu: float         # True anomaly [rad]
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor [a, e, i, Ω, ω, ν]."""
        return torch.tensor([
            self.a, self.e, self.i, self.Omega, self.omega, self.nu
        ], dtype=torch.float64)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "OrbitalElements":
        """Create from tensor."""
        return cls(
            a=tensor[0].item(),
            e=tensor[1].item(),
            i=tensor[2].item(),
            Omega=tensor[3].item(),
            omega=tensor[4].item(),
            nu=tensor[5].item(),
        )


def orbital_elements_to_cartesian(
    elements: Union[OrbitalElements, torch.Tensor],
    mu: float = 398600.4418,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert Keplerian elements to Cartesian state (q, p).
    
    Uses the standard transformation through the perifocal frame.
    
    Args:
        elements: Orbital elements (OrbitalElements or tensor [a,e,i,Ω,ω,ν])
        mu: Gravitational parameter (km³/s²)
        
    Returns:
        Tuple of (position, velocity) in ECI frame, each (3,)
    """
    if isinstance(elements, OrbitalElements):
        a = elements.a
        e = elements.e
        i = elements.i
        Omega = elements.Omega
        omega = elements.omega
        nu = elements.nu
    else:
        a, e, i, Omega, omega, nu = elements.tolist()
    
    # Semi-latus rectum
    p = a * (1 - e**2)
    
    # Distance
    r = p / (1 + e * np.cos(nu))
    
    # Position in perifocal frame
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0])
    
    # Velocity in perifocal frame
    v_pf = np.sqrt(mu / p) * np.array([
        -np.sin(nu),
        e + np.cos(nu),
        0
    ])
    
    # Rotation matrices
    R_Omega = np.array([
        [np.cos(Omega), -np.sin(Omega), 0],
        [np.sin(Omega), np.cos(Omega), 0],
        [0, 0, 1]
    ])
    
    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    R_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: ECI <- Perifocal
    R = R_Omega @ R_i @ R_omega
    
    # Transform to ECI
    r_eci = R @ r_pf
    v_eci = R @ v_pf
    
    return torch.tensor(r_eci, dtype=torch.float64), torch.tensor(v_eci, dtype=torch.float64)


def cartesian_to_orbital_elements(
    q: torch.Tensor,
    p: torch.Tensor,
    mu: float = 398600.4418,
) -> OrbitalElements:
    """
    Convert Cartesian state to Keplerian elements.
    
    Args:
        q: Position vector (3,)
        p: Velocity vector (3,)
        mu: Gravitational parameter
        
    Returns:
        OrbitalElements
    """
    q_np = q.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    
    # Specific angular momentum
    h = np.cross(q_np, p_np)
    h_mag = np.linalg.norm(h)
    
    # Node vector
    n = np.cross([0, 0, 1], h)
    n_mag = np.linalg.norm(n)
    
    # Eccentricity vector
    r = np.linalg.norm(q_np)
    v = np.linalg.norm(p_np)
    
    e_vec = ((v**2 - mu/r) * q_np - np.dot(q_np, p_np) * p_np) / mu
    e = np.linalg.norm(e_vec)
    
    # Specific energy
    energy = v**2 / 2 - mu / r
    
    # Semi-major axis
    if abs(e - 1.0) > 1e-10:
        a = -mu / (2 * energy)
    else:
        a = float('inf')  # Parabolic
    
    # Inclination
    i = np.arccos(h[2] / h_mag)
    
    # RAAN
    if n_mag > 1e-10:
        Omega = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0.0
    
    # Argument of periapsis
    if n_mag > 1e-10 and e > 1e-10:
        omega = np.arccos(np.dot(n, e_vec) / (n_mag * e))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0.0
    
    # True anomaly
    if e > 1e-10:
        nu = np.arccos(np.dot(e_vec, q_np) / (e * r))
        if np.dot(q_np, p_np) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0.0
    
    return OrbitalElements(
        a=a, e=e, i=i, Omega=Omega, omega=omega, nu=nu
    )


class SyntheticCatalog:
    """
    Generate synthetic orbital catalog with controlled distributions.
    
    Supports:
    - Training regime: Low inclination (equatorial)
    - Test regime: High inclination (polar)
    - Conjunction injection for rare events
    
    Args:
        mu: Gravitational parameter
        R_earth: Earth radius for altitude calculations
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        mu: float = 398600.4418,
        R_earth: float = 6378.137,
        seed: Optional[int] = None,
    ):
        self.mu = mu
        self.R_earth = R_earth
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate_leo_orbit(
        self,
        altitude_range: Tuple[float, float] = (400, 2000),
        inclination_range: Tuple[float, float] = (0, np.pi/9),  # 0-20°
        eccentricity_max: float = 0.1,
    ) -> OrbitalElements:
        """
        Generate a single LEO orbit.
        
        Args:
            altitude_range: (min, max) altitude in km
            inclination_range: (min, max) inclination in radians
            eccentricity_max: Maximum eccentricity
            
        Returns:
            OrbitalElements
        """
        # Semi-major axis from altitude
        alt = np.random.uniform(*altitude_range)
        a = self.R_earth + alt
        
        # Eccentricity (peaked near circular)
        e = np.random.beta(1, 5) * eccentricity_max
        
        # Inclination
        i = np.random.uniform(*inclination_range)
        
        # Other angles uniform
        Omega = np.random.uniform(0, 2 * np.pi)
        omega = np.random.uniform(0, 2 * np.pi)
        nu = np.random.uniform(0, 2 * np.pi)
        
        return OrbitalElements(a=a, e=e, i=i, Omega=Omega, omega=omega, nu=nu)
    
    def generate_training_set(
        self,
        n_objects: int,
        inclination_range: Tuple[float, float] = (0, np.radians(20)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training set (equatorial orbits).
        
        Args:
            n_objects: Number of orbital objects
            inclination_range: Inclination range (default 0-20°)
            
        Returns:
            Tuple of (positions, velocities) each (N, 3)
        """
        positions = []
        velocities = []
        
        for _ in range(n_objects):
            elements = self.generate_leo_orbit(inclination_range=inclination_range)
            q, p = orbital_elements_to_cartesian(elements, self.mu)
            positions.append(q)
            velocities.append(p)
        
        return torch.stack(positions), torch.stack(velocities)
    
    def generate_test_set(
        self,
        n_objects: int,
        inclination_range: Tuple[float, float] = (np.radians(60), np.radians(90)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate test set (polar orbits - rare regime).
        
        Args:
            n_objects: Number of orbital objects
            inclination_range: Inclination range (default 60-90°)
            
        Returns:
            Tuple of (positions, velocities) each (N, 3)
        """
        return self.generate_training_set(n_objects, inclination_range)
    
    def inject_conjunction(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n_conjunctions: int = 5,
        min_separation: float = 1.0,  # km
        max_separation: float = 50.0,  # km
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inject close-approach scenarios into the catalog.
        
        Modifies some orbital pairs to create conjunction events.
        
        Args:
            q: Positions (N, 3)
            p: Velocities (N, 3)
            n_conjunctions: Number of conjunction pairs to create
            min_separation: Minimum separation distance
            max_separation: Maximum separation distance
            
        Returns:
            Modified (q, p) and conjunction_pairs tensor
        """
        N = q.shape[0]
        q_new = q.clone()
        p_new = p.clone()
        
        conjunction_pairs = []
        
        for _ in range(min(n_conjunctions, N // 2)):
            # Select random pair
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            while j == i:
                j = np.random.randint(0, N)
            
            # Create close approach
            separation = np.random.uniform(min_separation, max_separation)
            direction = torch.randn(3, dtype=torch.float64)
            direction = direction / torch.norm(direction)
            
            # Move object j close to object i
            q_new[j] = q_new[i] + separation * direction
            
            # Adjust velocity to create crossing trajectory
            rel_vel = p_new[i] - p_new[j]
            # Add component toward object i
            p_new[j] = p_new[j] + 0.1 * torch.randn(3, dtype=torch.float64)
            
            conjunction_pairs.append([i, j])
        
        conjunction_pairs = torch.tensor(conjunction_pairs, dtype=torch.long)
        
        return q_new, p_new, conjunction_pairs
    
    def generate_dataset(
        self,
        n_train: int = 200,
        n_test: int = 50,
        n_conjunctions_train: int = 10,
        n_conjunctions_test: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate complete training and test datasets.
        
        Args:
            n_train: Number of training objects
            n_test: Number of test objects
            n_conjunctions_train: Conjunctions in training set
            n_conjunctions_test: Conjunctions in test set
            
        Returns:
            Dict with train/test positions, velocities, and labels
        """
        # Training set (equatorial)
        q_train, p_train = self.generate_training_set(n_train)
        q_train, p_train, conj_train = self.inject_conjunction(
            q_train, p_train, n_conjunctions_train
        )
        
        # Test set (polar - rare regime)
        q_test, p_test = self.generate_test_set(n_test)
        q_test, p_test, conj_test = self.inject_conjunction(
            q_test, p_test, n_conjunctions_test
        )
        
        # Create labels (1 if involved in conjunction, 0 otherwise)
        train_labels = torch.zeros(n_train, dtype=torch.float64)
        if len(conj_train) > 0:
            train_labels[conj_train[:, 0]] = 1.0
            train_labels[conj_train[:, 1]] = 1.0
        
        test_labels = torch.zeros(n_test, dtype=torch.float64)
        if len(conj_test) > 0:
            test_labels[conj_test[:, 0]] = 1.0
            test_labels[conj_test[:, 1]] = 1.0
        
        return {
            "train_q": q_train,
            "train_p": p_train,
            "train_labels": train_labels,
            "train_conjunctions": conj_train,
            "test_q": q_test,
            "test_p": p_test,
            "test_labels": test_labels,
            "test_conjunctions": conj_test,
        }
    
    def generate_trajectory_dataset(
        self,
        n_objects: int,
        n_steps: int,
        dt: float = 60.0,
        inclination_range: Tuple[float, float] = (0, np.radians(20)),
        include_j2: bool = False,
        j2: float = 1.08263e-3,
        r_eq: float = 6378.137,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate dataset with ground truth trajectories.
        
        Uses numerical integration for ground truth.
        
        Args:
            n_objects: Number of objects
            n_steps: Number of time steps
            dt: Time step in seconds
            inclination_range: Inclination range
            include_j2: If True, include J2 perturbation in ground truth
            j2: J2 coefficient (Earth default: 1.08263e-3)
            r_eq: Equatorial radius (km)
            
        Returns:
            Dict with initial states and trajectories
        """
        from ..integrators import StormerVerlet
        from ..models.hamiltonian import TwoBodyHamiltonian, KeplerianHamiltonian
        
        # Generate initial states
        q0, p0 = self.generate_training_set(n_objects, inclination_range)
        
        # Choose Hamiltonian based on J2 flag
        if include_j2:
            hamiltonian = KeplerianHamiltonian(mu=self.mu, J2=j2, R_eq=r_eq, include_J2=True)
        else:
            hamiltonian = TwoBodyHamiltonian(mu=self.mu)
            
        integrator = StormerVerlet(dt=dt)
        
        q_traj, p_traj = integrator.integrate(
            q0, p0, hamiltonian, n_steps, store_trajectory=True
        )
        
        return {
            "q_initial": q0,
            "p_initial": p0,
            "q_trajectory": q_traj,
            "p_trajectory": p_traj,
            "dt": dt,
            "n_steps": n_steps,
            "include_j2": include_j2,
        }


class ConjunctionDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for conjunction prediction.
    
    Each sample contains:
    - Initial state (q, p)
    - Target: minimum separation distance over horizon
    - Label: binary conjunction indicator
    """
    
    def __init__(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        targets: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.q = q
        self.p = p
        self.targets = targets
        self.labels = labels
    
    def __len__(self):
        return self.q.shape[0]
    
    def __getitem__(self, idx):
        return {
            "q": self.q[idx],
            "p": self.p[idx],
            "target": self.targets[idx],
            "label": self.labels[idx],
        }


def create_dataloader(
    dataset: ConjunctionDataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for training."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
