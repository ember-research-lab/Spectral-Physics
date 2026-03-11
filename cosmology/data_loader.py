#!/usr/bin/env python3
"""
Unified data loading for cosmology analyses.

Handles both real CosmicFlows-4++ data (CF4pp_mean_std_grids.npz, 161 MB)
and synthetic test data for CI environments without the large data file.

The CF4++ grid is a 128^3 density and velocity field covering a
1000 Mpc box centered on the Milky Way.

Book reference: Chapter 37 -- Cosmology
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# CF4++ grid constants
CF4_GRID_N = 128
CF4_BOX_MPC = 1000.0
CF4_CELL_MPC = CF4_BOX_MPC / CF4_GRID_N  # 7.8125 Mpc per cell

# Default data path (relative to this file)
DEFAULT_CF4_PATH = Path(__file__).parent / "data" / "CF4pp_mean_std_grids.npz"


def load_cf4_grid(
    filepath: Optional[str] = None,
    delta_threshold: float = 0.0,
    max_points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CF4++ grid data and extract overdense cells as point cloud.

    The CF4++ file contains 128^3 grids of density contrast and 3D velocity.
    We select cells above a density threshold and return their physical
    positions, velocities, and density contrast values.

    Parameters
    ----------
    filepath : str or None
        Path to CF4pp_mean_std_grids.npz. If None, uses default data path.
    delta_threshold : float
        Minimum density contrast to include a cell.
    max_points : int
        Maximum number of cells (subsamples highest-density cells).

    Returns
    -------
    positions : (N, 3) array
        Physical positions in Mpc, centered at origin.
    velocities : (N, 3) array
        Peculiar velocities in km/s.
    densities : (N,) array
        Density contrast values.
    """
    if filepath is None:
        filepath = str(DEFAULT_CF4_PATH)

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"CF4++ data not found at {filepath}. "
            "Download from the Extragalactic Distance Database "
            "(https://edd.ifa.hawaii.edu/). "
            "See cosmology/data/README.md for instructions."
        )

    data = np.load(filepath)
    d_mean = data["d_mean_CF4pp"]  # (128, 128, 128) density contrast
    v_mean = data["v_mean_CF4pp"]  # (3, 128, 128, 128) velocity field

    # Select cells above threshold
    mask = d_mean > delta_threshold
    ix, iy, iz = np.where(mask)
    n_cells = len(ix)

    # Convert grid indices to physical positions (cell centers)
    # Grid spans [-L/2, +L/2] in each dimension
    positions = np.column_stack([
        ix * CF4_CELL_MPC - CF4_BOX_MPC / 2 + CF4_CELL_MPC / 2,
        iy * CF4_CELL_MPC - CF4_BOX_MPC / 2 + CF4_CELL_MPC / 2,
        iz * CF4_CELL_MPC - CF4_BOX_MPC / 2 + CF4_CELL_MPC / 2,
    ])

    velocities = np.column_stack([
        v_mean[0][mask],
        v_mean[1][mask],
        v_mean[2][mask],
    ])

    densities = d_mean[mask]

    # Subsample to highest-density cells if needed
    if n_cells > max_points:
        top_idx = np.argsort(densities)[-max_points:]
        positions = positions[top_idx]
        velocities = velocities[top_idx]
        densities = densities[top_idx]

    return positions, velocities, densities


def load_cf4_subgrid_fields(
    filepath: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the full CF4++ 128^3 density and radial velocity fields.

    Used by scripts that analyze subgrid statistics (density-lambda1
    correlation, Hubble anticorrelation, non-tautological test).

    Parameters
    ----------
    filepath : str or None
        Path to CF4pp_mean_std_grids.npz.

    Returns
    -------
    delta : (128, 128, 128) array
        Density contrast field.
    v_rad : (128, 128, 128) array
        Radial peculiar velocity field (km/s).
    """
    if filepath is None:
        filepath = str(DEFAULT_CF4_PATH)

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"CF4++ data not found at {filepath}. "
            "See cosmology/data/README.md for download instructions."
        )

    data = np.load(filepath)
    delta = data["d_mean_CF4pp"]
    v_rad = data["vr_mean_CF4pp"]

    return delta, v_rad


def create_synthetic_universe(
    n_grid: int = 32,
    box_mpc: float = 200.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a synthetic universe for testing without CF4++ data.

    Generates a 3D density field with:
    - A density gradient (high density at one corner, low at opposite)
    - Gaussian perturbations for structure
    - Velocity field correlated with density gradient (infall toward overdensity)

    This produces galaxy positions that exhibit the expected
    spectral cosmology correlations (rho-lambda1 positive, lambda1-H negative)
    at reduced significance compared to real data.

    Parameters
    ----------
    n_grid : int
        Grid size per dimension.
    box_mpc : float
        Box side length in Mpc.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    positions : (N, 3) array
        Physical positions in Mpc.
    velocities : (N, 3) array
        Peculiar velocities in km/s.
    densities : (N,) array
        Density contrast values.
    """
    rng = np.random.default_rng(seed)
    cell = box_mpc / n_grid

    # Create 3D coordinate arrays
    x = np.arange(n_grid) * cell - box_mpc / 2 + cell / 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    # Density: gradient + Gaussian noise
    # Large-scale gradient: denser toward positive corner
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r_max = np.sqrt(3) * box_mpc / 2
    gradient = 1.5 * (1 - r / r_max)  # 1.5 at center, ~0 at corners

    # Small-scale perturbations
    noise = rng.normal(0, 0.3, (n_grid, n_grid, n_grid))
    delta = gradient + noise

    # Velocity: objects fall toward high-density regions
    # Approximate with gradient of potential (negative density gradient)
    # Radial velocity: outward in voids, inward near clusters
    v_scale = 300.0  # km/s amplitude
    dx = np.gradient(delta, cell, axis=0)
    dy = np.gradient(delta, cell, axis=1)
    dz = np.gradient(delta, cell, axis=2)
    grad_mag = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-10
    vx = -v_scale * dx / grad_mag * np.abs(delta)
    vy = -v_scale * dy / grad_mag * np.abs(delta)
    vz = -v_scale * dz / grad_mag * np.abs(delta)

    # Add Hubble flow (outward velocity proportional to distance)
    H0 = 67.4  # km/s/Mpc
    vx += H0 * X * 0.01  # Small perturbation
    vy += H0 * Y * 0.01
    vz += H0 * Z * 0.01

    # Flatten to point cloud
    positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    velocities = np.column_stack([vx.ravel(), vy.ravel(), vz.ravel()])
    densities = delta.ravel()

    return positions, velocities, densities


def create_synthetic_subgrid_fields(
    n_grid: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic density and radial velocity fields for subgrid analyses.

    Mimics the CF4++ 128^3 format at smaller scale for testing.

    Parameters
    ----------
    n_grid : int
        Grid size per dimension.
    seed : int
        Random seed.

    Returns
    -------
    delta : (n_grid, n_grid, n_grid) array
        Density contrast.
    v_rad : (n_grid, n_grid, n_grid) array
        Radial peculiar velocity (km/s).
    """
    rng = np.random.default_rng(seed)

    # Density field with large-scale gradient + noise
    x = np.linspace(-1, 1, n_grid)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)

    # Gradient: center is overdense
    delta = 2.0 * np.exp(-r**2 / 0.5) + rng.normal(0, 0.3, (n_grid, n_grid, n_grid))

    # Radial velocity: anticorrelated with density
    # Overdense regions have infall (negative v_r), voids expand (positive v_r)
    v_rad = -200.0 * delta + rng.normal(0, 50, (n_grid, n_grid, n_grid))

    return delta, v_rad


if __name__ == "__main__":
    print("Data Loader -- Cosmology Module")
    print("=" * 50)

    # Check for real data
    if DEFAULT_CF4_PATH.exists():
        print(f"CF4++ data found at {DEFAULT_CF4_PATH}")
        positions, velocities, densities = load_cf4_grid()
        print(f"  Loaded {len(positions)} cells")
        print(f"  Position range: [{positions.min():.1f}, {positions.max():.1f}] Mpc")
        print(f"  Density range: [{densities.min():.3f}, {densities.max():.3f}]")
    else:
        print(f"CF4++ data not found at {DEFAULT_CF4_PATH}")
        print("Using synthetic data for demonstration.\n")

        positions, velocities, densities = create_synthetic_universe()
        print(f"Synthetic universe: {len(positions)} cells")
        print(f"  Position range: [{positions.min():.1f}, {positions.max():.1f}] Mpc")
        print(f"  Density range: [{densities.min():.3f}, {densities.max():.3f}]")
        print(f"  Velocity range: [{velocities.min():.1f}, {velocities.max():.1f}] km/s")
