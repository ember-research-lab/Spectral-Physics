#!/usr/bin/env python3
"""
Universe eigenstate computation.

The observable universe can be decomposed into spherical harmonic modes.
The state is overwhelmingly dominated by the ground state (monopole):

    |psi> ~ 0.999998|0,0> + 0.0021|1,m> + O(1e-5)|2,m>

The dipole coefficient comes from the bulk flow velocity (~630 km/s
toward Shapley), and the quadrupole from CMB anisotropy measurements.

This gives 99.9996% ground-state occupation -- the universe is
spectrally cold, sitting very near the ground state of its own Laplacian.

Book reference: Chapter 37 -- Cosmology
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Physical constants
C_LIGHT = 299792.458  # km/s
H0 = 67.4  # km/s/Mpc (Planck 2018)
V_BULK_OBSERVED = 630.0  # km/s bulk flow toward Shapley (Kashlinsky+2010)
CMB_QUADRUPOLE_DT_T = 1e-5  # delta T / T for quadrupole


@dataclass
class UniverseEigenstate:
    """Eigenstate decomposition of the observable universe."""

    c_0: float  # Ground state (monopole) amplitude
    c_1: float  # Dipole amplitude (from bulk flow)
    c_2: float  # Quadrupole amplitude (from CMB)
    ground_state_fraction: float  # |c_0|^2 / sum(|c_k|^2)
    v_bulk: float  # Bulk flow velocity used (km/s)

    def __str__(self) -> str:
        return (
            f"Universe Eigenstate Decomposition\n"
            f"{'=' * 45}\n"
            f"  c_0 (monopole)   = {self.c_0:.6f}\n"
            f"  c_1 (dipole)     = {self.c_1:.6f}\n"
            f"  c_2 (quadrupole) = {self.c_2:.2e}\n"
            f"  Ground state     = {self.ground_state_fraction * 100:.4f}%\n"
            f"  Bulk flow used   = {self.v_bulk:.0f} km/s"
        )


def compute_eigenstate(
    v_bulk: Optional[float] = None,
    quadrupole_dt_t: Optional[float] = None,
) -> UniverseEigenstate:
    """
    Compute the universe eigenstate from observed bulk flow and CMB data.

    The eigenstate coefficients are:
        c_0 = 1.0                    (background, normalized)
        c_1 = v_bulk / c             (dipole from bulk flow)
        c_2 = quadrupole_dT/T        (quadrupole from CMB)

    Parameters
    ----------
    v_bulk : float or None
        Bulk flow velocity in km/s. Defaults to 630 km/s.
    quadrupole_dt_t : float or None
        CMB quadrupole dT/T. Defaults to 1e-5.

    Returns
    -------
    UniverseEigenstate
        Eigenstate decomposition with amplitudes and ground state fraction.
    """
    if v_bulk is None:
        v_bulk = V_BULK_OBSERVED
    if quadrupole_dt_t is None:
        quadrupole_dt_t = CMB_QUADRUPOLE_DT_T

    c_0 = 1.0
    c_1 = v_bulk / C_LIGHT  # ~ 0.0021 for 630 km/s
    c_2 = quadrupole_dt_t   # ~ 1e-5

    # Ground state fraction (normalized probability)
    norm_sq = c_0**2 + c_1**2 + c_2**2
    ground_state_fraction = c_0**2 / norm_sq

    return UniverseEigenstate(
        c_0=c_0,
        c_1=c_1,
        c_2=c_2,
        ground_state_fraction=ground_state_fraction,
        v_bulk=v_bulk,
    )


def compute_eigenstate_from_velocities(
    velocities: np.ndarray,
) -> UniverseEigenstate:
    """
    Compute eigenstate from a velocity field (e.g., CF4++ data).

    The bulk flow is the mean velocity vector magnitude.

    Parameters
    ----------
    velocities : (N, 3) array
        3D peculiar velocities in km/s.

    Returns
    -------
    UniverseEigenstate
    """
    mean_velocity = np.mean(velocities, axis=0)
    v_bulk = np.linalg.norm(mean_velocity)
    return compute_eigenstate(v_bulk=v_bulk)


def eigenstate_energy_levels(
    n_levels: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the first n energy levels and their occupation fractions.

    For the universe eigenstate, higher modes are exponentially suppressed.
    This uses the observed values for l=0,1,2 and theoretical estimates
    for higher modes.

    Parameters
    ----------
    n_levels : int
        Number of levels to compute.

    Returns
    -------
    levels : (n_levels,) array of int
        Angular momentum quantum numbers l = 0, 1, ...
    occupations : (n_levels,) array
        |c_l|^2 for each level.
    """
    state = compute_eigenstate()

    levels = np.arange(n_levels)
    occupations = np.zeros(n_levels)
    occupations[0] = state.c_0**2
    if n_levels > 1:
        occupations[1] = state.c_1**2
    if n_levels > 2:
        occupations[2] = state.c_2**2

    # Higher modes decay exponentially (theoretical estimate)
    for l in range(3, n_levels):
        occupations[l] = state.c_2**2 * 10**(-(l - 2))

    # Normalize
    occupations /= occupations.sum()

    return levels, occupations


if __name__ == "__main__":
    print("Universe Eigenstate -- Cosmology Module")
    print("=" * 50)

    state = compute_eigenstate()
    print(state)

    print(f"\nKey result: {state.ground_state_fraction * 100:.4f}% ground state")
    print(f"The universe is spectrally cold.")

    print(f"\nEnergy levels:")
    levels, occupations = eigenstate_energy_levels(5)
    for l, occ in zip(levels, occupations):
        print(f"  l={l}: {occ:.2e}")
