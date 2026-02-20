"""
TLE (Two-Line Element) Parser for Real Orbital Data.

Parses TLE format used by NORAD/SpaceTrack and converts
to Cartesian states using SGP4 propagator.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings


@dataclass
class TLEData:
    """
    Parsed TLE data structure.
    
    Attributes:
        name: Satellite name
        norad_id: NORAD catalog number
        epoch: Epoch time
        mean_motion: Mean motion (rev/day)
        eccentricity: Eccentricity
        inclination: Inclination (degrees)
        raan: Right ascension of ascending node (degrees)
        arg_perigee: Argument of perigee (degrees)
        mean_anomaly: Mean anomaly (degrees)
        bstar: Drag term
    """
    name: str
    norad_id: int
    epoch_year: int
    epoch_day: float
    mean_motion: float
    eccentricity: float
    inclination: float
    raan: float
    arg_perigee: float
    mean_anomaly: float
    bstar: float = 0.0
    
    def to_cartesian(
        self,
        minutes_from_epoch: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert to Cartesian state using SGP4.
        
        Requires sgp4 package: pip install sgp4
        
        Args:
            minutes_from_epoch: Time offset from TLE epoch
            
        Returns:
            Tuple of (position_km, velocity_km_s)
        """
        try:
            from sgp4.api import Satrec, WGS72
            from sgp4.api import jday
        except ImportError:
            raise ImportError(
                "sgp4 package required for TLE propagation. "
                "Install with: pip install sgp4"
            )
        
        # Create SGP4 satellite object
        satrec = Satrec()
        satrec.sgp4init(
            WGS72,
            'i',  # Improved mode
            self.norad_id,
            self._epoch_to_jd() - 2433281.5,  # Epoch days since 1949
            self.bstar,
            0.0,  # ndot (not used in SGP4)
            0.0,  # nddot (not used in SGP4)
            self.eccentricity,
            np.radians(self.arg_perigee),
            np.radians(self.inclination),
            np.radians(self.mean_anomaly),
            self.mean_motion * 2 * np.pi / 1440.0,  # rad/min
            np.radians(self.raan),
        )
        
        # Propagate
        jd, fr = self._epoch_to_jd_fr(minutes_from_epoch)
        error, r, v = satrec.sgp4(jd, fr)
        
        if error != 0:
            warnings.warn(f"SGP4 propagation error code: {error}")
        
        return (
            torch.tensor(r, dtype=torch.float64),
            torch.tensor(v, dtype=torch.float64),
        )
    
    def _epoch_to_jd(self) -> float:
        """Convert TLE epoch to Julian date."""
        year = self.epoch_year
        if year < 57:
            year += 2000
        else:
            year += 1900
        
        # January 1 of epoch year
        from datetime import datetime
        jan1 = datetime(year, 1, 1)
        
        # Julian date of Jan 1
        jd_jan1 = (
            367 * year
            - int(7 * (year + int((1 + 9) / 12)) / 4)
            + int(275 * 1 / 9)
            + 1
            + 1721013.5
        )
        
        return jd_jan1 + self.epoch_day - 1
    
    def _epoch_to_jd_fr(
        self,
        minutes_offset: float = 0.0,
    ) -> Tuple[float, float]:
        """Get Julian date as (day, fraction) for SGP4."""
        jd = self._epoch_to_jd() + minutes_offset / 1440.0
        jd_int = int(jd)
        jd_fr = jd - jd_int
        return jd_int, jd_fr


class TLEParser:
    """
    Parse TLE data from text.
    
    Supports standard two-line format and three-line format
    (with satellite name).
    """
    
    def __init__(self):
        pass
    
    def parse(self, tle_text: str) -> List[TLEData]:
        """
        Parse TLE text into list of TLEData objects.
        
        Args:
            tle_text: Raw TLE text (can contain multiple entries)
            
        Returns:
            List of TLEData objects
        """
        lines = [l.strip() for l in tle_text.strip().split('\n') if l.strip()]
        
        tles = []
        i = 0
        
        while i < len(lines):
            # Check if this is a name line (doesn't start with 1 or 2)
            if not lines[i].startswith('1') and not lines[i].startswith('2'):
                name = lines[i]
                i += 1
            else:
                name = "UNKNOWN"
            
            if i + 1 >= len(lines):
                break
            
            line1 = lines[i]
            line2 = lines[i + 1]
            i += 2
            
            try:
                tle = self._parse_lines(name, line1, line2)
                tles.append(tle)
            except Exception as e:
                warnings.warn(f"Failed to parse TLE: {e}")
                continue
        
        return tles
    
    def _parse_lines(
        self,
        name: str,
        line1: str,
        line2: str,
    ) -> TLEData:
        """Parse a single TLE entry."""
        # Line 1 parsing
        norad_id = int(line1[2:7])
        epoch_year = int(line1[18:20])
        epoch_day = float(line1[20:32])
        
        # B* drag term (with implied decimal point)
        bstar_mantissa = line1[53:59]
        bstar_exp = line1[59:61]
        if bstar_mantissa.strip():
            bstar = float(f"0.{bstar_mantissa.strip()}") * 10 ** int(bstar_exp)
        else:
            bstar = 0.0
        
        # Line 2 parsing
        inclination = float(line2[8:16])
        raan = float(line2[17:25])
        
        # Eccentricity (implied leading decimal)
        ecc_str = line2[26:33]
        eccentricity = float(f"0.{ecc_str}")
        
        arg_perigee = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion = float(line2[52:63])
        
        return TLEData(
            name=name,
            norad_id=norad_id,
            epoch_year=epoch_year,
            epoch_day=epoch_day,
            mean_motion=mean_motion,
            eccentricity=eccentricity,
            inclination=inclination,
            raan=raan,
            arg_perigee=arg_perigee,
            mean_anomaly=mean_anomaly,
            bstar=bstar,
        )
    
    def parse_file(self, filepath: str) -> List[TLEData]:
        """Parse TLE data from file."""
        with open(filepath, 'r') as f:
            return self.parse(f.read())
    
    def to_cartesian_batch(
        self,
        tles: List[TLEData],
        minutes_from_epoch: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert list of TLEs to batched Cartesian states.
        
        Args:
            tles: List of TLEData objects
            minutes_from_epoch: Time offset
            
        Returns:
            Tuple of (positions, velocities) each (N, 3)
        """
        positions = []
        velocities = []
        
        for tle in tles:
            try:
                r, v = tle.to_cartesian(minutes_from_epoch)
                positions.append(r)
                velocities.append(v)
            except Exception as e:
                warnings.warn(f"Skipping TLE {tle.name}: {e}")
                continue
        
        if not positions:
            return torch.zeros(0, 3), torch.zeros(0, 3)
        
        return torch.stack(positions), torch.stack(velocities)


# Example TLE for testing
EXAMPLE_TLE = """ISS (ZARYA)
1 25544U 98067A   21275.52628472  .00001878  00000-0  42323-4 0  9993
2 25544  51.6442 208.5455 0003632  46.5765  52.5844 15.48919419305837"""


def load_example_tle() -> TLEData:
    """Load example ISS TLE for testing."""
    parser = TLEParser()
    tles = parser.parse(EXAMPLE_TLE)
    return tles[0] if tles else None
