"""
TLE Data Loader and Trajectory Generator

Loads real satellite TLE (Two-Line Element) data from CelesTrak and converts
to trajectory datasets for training H-GNN models.

TLE Format:
    Line 0: Satellite Name
    Line 1: 1 NNNNNC NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
    Line 2: 2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN

The SGP4 propagator gives position/velocity in TEME (True Equator Mean Equinox) frame.
These are converted to the (q, p) state representation for H-GNN.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    print("Warning: sgp4 not installed. Run: pip install sgp4")


@dataclass
class SatelliteInfo:
    """Information about a satellite from TLE."""
    name: str
    norad_id: int
    line1: str
    line2: str
    epoch_year: int
    epoch_day: float
    inclination: float  # degrees
    eccentricity: float
    mean_motion: float  # revolutions per day
    

def parse_tle_file(filepath: str) -> List[SatelliteInfo]:
    """
    Parse a TLE file containing multiple satellites.
    
    Args:
        filepath: Path to .tle file
        
    Returns:
        List of SatelliteInfo objects
    """
    satellites = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # TLE format: 3 lines per satellite (name, line1, line2)
    i = 0
    while i < len(lines) - 2:
        name = lines[i].strip()
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()
        
        # Validate TLE format
        if not line1.startswith('1 ') or not line2.startswith('2 '):
            i += 1
            continue
        
        try:
            # Parse key orbital elements
            norad_id = int(line1[2:7])
            epoch_year = int(line1[18:20])
            epoch_year = epoch_year + (2000 if epoch_year < 57 else 1900)
            epoch_day = float(line1[20:32])
            
            inclination = float(line2[8:16])
            eccentricity = float('0.' + line2[26:33])
            mean_motion = float(line2[52:63])
            
            satellites.append(SatelliteInfo(
                name=name,
                norad_id=norad_id,
                line1=line1,
                line2=line2,
                epoch_year=epoch_year,
                epoch_day=epoch_day,
                inclination=inclination,
                eccentricity=eccentricity,
                mean_motion=mean_motion,
            ))
        except (ValueError, IndexError) as e:
            pass  # Skip malformed entries
        
        i += 3
    
    return satellites


def propagate_satellite(
    sat_info: SatelliteInfo,
    start_time: datetime,
    duration_minutes: int = 1440,  # 24 hours
    step_minutes: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate satellite trajectory using SGP4.
    
    Args:
        sat_info: Satellite TLE information
        start_time: Start datetime for propagation
        duration_minutes: Total duration in minutes
        step_minutes: Time step in minutes
        
    Returns:
        positions: (T, 3) array of positions in km
        velocities: (T, 3) array of velocities in km/s
    """
    if not SGP4_AVAILABLE:
        raise ImportError("sgp4 library required. Install with: pip install sgp4")
    
    satellite = Satrec.twoline2rv(sat_info.line1, sat_info.line2)
    
    positions = []
    velocities = []
    
    for minute in range(0, duration_minutes, step_minutes):
        current_time = start_time + timedelta(minutes=minute)
        
        jd, fr = jday(
            current_time.year,
            current_time.month,
            current_time.day,
            current_time.hour,
            current_time.minute,
            current_time.second + current_time.microsecond / 1e6
        )
        
        error_code, position, velocity = satellite.sgp4(jd, fr)
        
        if error_code == 0:
            positions.append(position)
            velocities.append(velocity)
        else:
            # Propagation failed (e.g., decayed satellite)
            break
    
    return np.array(positions), np.array(velocities)


def generate_trajectory_dataset(
    tle_file: str,
    n_satellites: int = 10,
    duration_minutes: int = 1440,
    step_minutes: int = 1,
    start_time: Optional[datetime] = None,
    filter_leo: bool = True,
    min_altitude_km: float = 200,
    max_altitude_km: float = 2000,
) -> Dict[str, Any]:
    """
    Generate trajectory dataset from TLE file.
    
    Args:
        tle_file: Path to TLE file
        n_satellites: Number of satellites to include
        duration_minutes: Trajectory duration in minutes
        step_minutes: Time step in minutes
        start_time: Start time (default: now)
        filter_leo: If True, only select LEO satellites
        min_altitude_km: Minimum altitude for filtering
        max_altitude_km: Maximum altitude for filtering
        
    Returns:
        Dictionary with trajectory data and metadata
    """
    if start_time is None:
        start_time = datetime.utcnow()
    
    # Parse TLE file
    satellites = parse_tle_file(tle_file)
    print(f"Parsed {len(satellites)} satellites from TLE file")
    
    # Filter for LEO if requested
    if filter_leo:
        # LEO satellites have mean motion > 11 rev/day (period < 130 minutes)
        # This corresponds roughly to altitude < 2000 km
        leo_satellites = [
            s for s in satellites 
            if s.mean_motion > 11 and s.eccentricity < 0.1
        ]
        print(f"Filtered to {len(leo_satellites)} LEO satellites")
        satellites = leo_satellites
    
    # Select random subset
    if len(satellites) > n_satellites:
        indices = np.random.choice(len(satellites), n_satellites, replace=False)
        satellites = [satellites[i] for i in indices]
    
    # Generate trajectories
    all_positions = []
    all_velocities = []
    sat_names = []
    
    for sat in satellites:
        try:
            positions, velocities = propagate_satellite(
                sat, start_time, duration_minutes, step_minutes
            )
            
            if len(positions) > 0:
                all_positions.append(positions)
                all_velocities.append(velocities)
                sat_names.append(sat.name)
                print(f"  {sat.name}: {len(positions)} points")
        except Exception as e:
            print(f"  {sat.name}: FAILED - {e}")
    
    # Stack into tensors
    # Shape: (n_satellites, T, 3)
    min_len = min(len(p) for p in all_positions)
    
    positions_array = np.stack([p[:min_len] for p in all_positions])
    velocities_array = np.stack([v[:min_len] for v in all_velocities])
    
    return {
        'positions': positions_array,
        'velocities': velocities_array,
        'satellite_names': sat_names,
        'n_satellites': len(sat_names),
        'n_timesteps': min_len,
        'step_minutes': step_minutes,
        'start_time': start_time.isoformat(),
        'duration_minutes': duration_minutes,
    }


def convert_to_hgnn_format(
    trajectory_data: Dict[str, Any],
    horizon: int = 50,
    stride: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert trajectory dataset to H-GNN training format.
    
    Args:
        trajectory_data: Output from generate_trajectory_dataset
        horizon: Number of steps to predict
        stride: Step between training samples
        
    Returns:
        q0: Initial positions (n_samples, n_particles, 3)
        p0: Initial velocities (n_samples, n_particles, 3)
        q_target: Target positions (n_samples, n_particles, 3)
        p_target: Target velocities (n_samples, n_particles, 3)
    """
    positions = trajectory_data['positions']  # (n_sat, T, 3)
    velocities = trajectory_data['velocities']  # (n_sat, T, 3)
    
    n_sat, T, _ = positions.shape
    
    q0_list = []
    p0_list = []
    q_target_list = []
    p_target_list = []
    
    for start_idx in range(0, T - horizon, stride):
        # Initial state: all satellites at time t
        q0 = positions[:, start_idx, :]  # (n_sat, 3)
        p0 = velocities[:, start_idx, :]  # (n_sat, 3)
        
        # Target state: all satellites at time t + horizon
        q_target = positions[:, start_idx + horizon, :]
        p_target = velocities[:, start_idx + horizon, :]
        
        q0_list.append(q0)
        p0_list.append(p0)
        q_target_list.append(q_target)
        p_target_list.append(p_target)
    
    q0 = torch.tensor(np.stack(q0_list), dtype=torch.float32)
    p0 = torch.tensor(np.stack(p0_list), dtype=torch.float32)
    q_target = torch.tensor(np.stack(q_target_list), dtype=torch.float32)
    p_target = torch.tensor(np.stack(p_target_list), dtype=torch.float32)
    
    return q0, p0, q_target, p_target


class TLEDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for TLE trajectory data.
    
    Example:
        dataset = TLEDataset('satellites.tle', n_satellites=10, horizon=50)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        for q0, p0, q_target, p_target in dataloader:
            # q0: (batch, n_particles, 3)
            # Train model...
    """
    
    def __init__(
        self,
        tle_file: str,
        n_satellites: int = 10,
        horizon: int = 50,
        duration_minutes: int = 1440,
        step_minutes: int = 1,
        stride: int = 10,
        cache_file: Optional[str] = None,
    ):
        self.horizon = horizon
        
        # Try to load from cache
        if cache_file and Path(cache_file).exists():
            print(f"Loading cached data from {cache_file}")
            cached = torch.load(cache_file)
            self.q0 = cached['q0']
            self.p0 = cached['p0']
            self.q_target = cached['q_target']
            self.p_target = cached['p_target']
            self.metadata = cached['metadata']
        else:
            # Generate from TLE
            print(f"Generating trajectories from {tle_file}")
            trajectory_data = generate_trajectory_dataset(
                tle_file,
                n_satellites=n_satellites,
                duration_minutes=duration_minutes,
                step_minutes=step_minutes,
            )
            
            self.q0, self.p0, self.q_target, self.p_target = convert_to_hgnn_format(
                trajectory_data, horizon=horizon, stride=stride
            )
            
            self.metadata = {
                'satellite_names': trajectory_data['satellite_names'],
                'n_satellites': trajectory_data['n_satellites'],
                'horizon': horizon,
                'stride': stride,
            }
            
            # Cache for future use
            if cache_file:
                print(f"Caching data to {cache_file}")
                torch.save({
                    'q0': self.q0,
                    'p0': self.p0,
                    'q_target': self.q_target,
                    'p_target': self.p_target,
                    'metadata': self.metadata,
                }, cache_file)
    
    def __len__(self):
        return len(self.q0)
    
    def __getitem__(self, idx):
        return (
            self.q0[idx],
            self.p0[idx],
            self.q_target[idx],
            self.p_target[idx],
        )


def quick_test():
    """Quick test of TLE loading and trajectory generation."""
    print("=" * 60)
    print("TLE Data Loader Test")
    print("=" * 60)
    
    # Example ISS TLE (may be outdated)
    iss_tle = SatelliteInfo(
        name="ISS (ZARYA)",
        norad_id=25544,
        line1="1 25544U 98067A   24101.53055556  .00016717  00000+0  10270-3 0  9991",
        line2="2 25544  51.6436 193.2652 0005044  94.4855 265.6651 15.50000000447885",
        epoch_year=2024,
        epoch_day=101.53,
        inclination=51.6436,
        eccentricity=0.0005044,
        mean_motion=15.5,
    )
    
    print(f"\nTest satellite: {iss_tle.name}")
    print(f"  NORAD ID: {iss_tle.norad_id}")
    print(f"  Inclination: {iss_tle.inclination}°")
    print(f"  Eccentricity: {iss_tle.eccentricity}")
    print(f"  Mean motion: {iss_tle.mean_motion} rev/day")
    
    if SGP4_AVAILABLE:
        # Propagate for 90 minutes (roughly one orbit)
        start = datetime(2024, 4, 10, 0, 0, 0)
        positions, velocities = propagate_satellite(iss_tle, start, duration_minutes=90, step_minutes=1)
        
        print(f"\nPropagated {len(positions)} timesteps")
        print(f"  Position range: {positions.min():.1f} to {positions.max():.1f} km")
        print(f"  Velocity range: {velocities.min():.3f} to {velocities.max():.3f} km/s")
        
        # Convert to torch
        q = torch.tensor(positions, dtype=torch.float32)
        p = torch.tensor(velocities, dtype=torch.float32)
        
        print(f"\nTensor shapes:")
        print(f"  q (positions): {q.shape}")
        print(f"  p (velocities): {p.shape}")
        
        # Verify orbital altitude
        r = torch.norm(q, dim=-1)
        altitude = r - 6371  # Earth radius in km
        print(f"\nOrbital altitude: {altitude.min():.1f} to {altitude.max():.1f} km")
    else:
        print("\nsgp4 not available. Install with: pip install sgp4")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    quick_test()
    
    # Test with actual TLE file if available
    tle_file = Path(__file__).parent.parent.parent / "satellites.tle"
    if tle_file.exists():
        print(f"\nLoading real TLE data from {tle_file}")
        satellites = parse_tle_file(str(tle_file))
        print(f"Found {len(satellites)} satellites")
        
        # Show first 5
        print("\nFirst 5 satellites:")
        for sat in satellites[:5]:
            print(f"  {sat.name}: NORAD {sat.norad_id}, inc={sat.inclination:.1f}°, e={sat.eccentricity:.4f}")
