from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TargetOrbit:
    # ---- Значения ----
    a: Optional[float] = None  # м
    e: Optional[float] = None
    inc_deg: Optional[float] = None  # °
    arg_periapsis_deg: Optional[float] = None  # °
    raan_deg: Optional[float] = None  # °

    # ---- Диапазоны ----
    a_bounds: Optional[tuple] = None  # м,
    e_bounds: Optional[tuple] = None  #
    inc_bounds_deg: Optional[tuple] = None  # °,
    arg_periapsis_bounds_deg: Optional[tuple] = None  # °
    raan_bounds_deg: Optional[tuple] = None  # °


def apply_target_orbit(phase, orbit: TargetOrbit) -> None:
    """Добавить boundary-constraints целевой орбиты на dm.Phase."""

    if orbit.a is not None:
        lo, hi = orbit.a_bounds
        phase.add_boundary_constraint(
            'orbit_a', loc='final',
            lower=orbit.a + lo,
            upper=orbit.a + hi,
            ref=orbit.a,
        )

    if orbit.e is not None:
        lo, hi = orbit.e_bounds
        phase.add_boundary_constraint(
            'orbit_e', loc='final',
            lower=max(orbit.e + lo, 0.0),
            upper=orbit.e + hi,
            ref=max(orbit.e, 0.01),
        )

    if orbit.inc_deg is not None:
        inc_rad = np.deg2rad(orbit.inc_deg)
        lo_rad = np.deg2rad(orbit.inc_bounds_deg[0])
        hi_rad = np.deg2rad(orbit.inc_bounds_deg[1])
        phase.add_boundary_constraint(
            'orbit_inc', loc='final',
            lower=max(inc_rad + lo_rad, 0.0),
            upper=inc_rad + hi_rad,
            ref=max(inc_rad, 0.01),
        )

    if orbit.arg_periapsis_deg is not None:
        w_rad = np.deg2rad(orbit.arg_periapsis_deg)
        lo_rad = np.deg2rad(orbit.arg_periapsis_bounds_deg[0])
        hi_rad = np.deg2rad(orbit.arg_periapsis_bounds_deg[1])
        phase.add_boundary_constraint(
            'orbit_arg_periapsis', loc='final',
            lower=w_rad + lo_rad,
            upper=w_rad + hi_rad,
            ref=np.pi,
        )

    if orbit.raan_deg is not None:
        raan_rad = np.deg2rad(orbit.raan_deg)
        lo_rad = np.deg2rad(orbit.raan_bounds_deg[0])
        hi_rad = np.deg2rad(orbit.raan_bounds_deg[1])
        phase.add_boundary_constraint(
            'orbit_raan', loc='final',
            lower=raan_rad + lo_rad,
            upper=raan_rad + hi_rad,
            ref=np.pi,
        )
