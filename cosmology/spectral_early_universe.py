#!/usr/bin/env python3
"""
Spectral early universe analysis.

Uses Tikhonov-regularized heat kernel retrodiction to predict
the mode structure of the early universe from the z=0 cosmic
Laplacian.

Key results:
    lambda_1 = 0.000187
    lambda_2 / lambda_1 = 7.8
    97.9% ground state at z=0
    3 modes recoverable at z=10
    13 spectral fixed points

IMPORTANT: Spectral time: tau_s = 3*H0^2*Delta_t
    (NOT Delta_t/tau_s = sqrt(3)*H0*Delta_t)

Book reference: Chapter 37 -- Cosmology
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .data_loader import DEFAULT_CF4_PATH, load_cf4_grid

# Physical constants
H0_KM_S_MPC = 67.4
H0_GYR_INV = H0_KM_S_MPC / 978.0
OMEGA_M = 0.315
SIGMA8 = 0.811
RHO_CRIT_0 = 2.775e11 * (H0_KM_S_MPC / 100)**2  # M_sun / Mpc^3
DELTA_CRIT = 1.686

# Spectral time: tau_s uses 3*H0^2 convention
LAMBDA1_PHYS = 3 * H0_GYR_INV**2  # Gyr^-2

CF4_BOX_MPC = 1000.0

# Lookback times (Gyr) for target redshifts
LOOKBACK_TIMES = {6: 12.80, 10: 13.18, 14: 13.40}


def construct_cosmic_laplacian(
    positions: np.ndarray,
    linking_length: float,
    sigma: Optional[float] = None,
) -> tuple:
    """
    Construct cosmic Laplacian from galaxy positions.

    Returns (L, W, lcc_indices) restricted to largest connected component.
    """
    N = len(positions)
    if sigma is None:
        sigma = linking_length / 3.0

    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=linking_length, output_type="ndarray")

    if len(pairs) == 0:
        raise ValueError(f"No edges found with linking_length={linking_length}")

    dists = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    weights = np.exp(-dists**2 / (2 * sigma**2))

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.concatenate([weights, weights])
    W = csr_matrix((data, (rows, cols)), shape=(N, N))

    degrees = np.array(W.sum(axis=1)).flatten()
    L = diags(degrees) - W

    # Restrict to largest connected component
    n_comp, labels = connected_components(W, directed=False)
    if n_comp > 1:
        comp_sizes = np.bincount(labels)
        largest = np.argmax(comp_sizes)
        lcc_idx = np.where(labels == largest)[0]
        W = W[lcc_idx][:, lcc_idx]
        degrees_lcc = np.array(W.sum(axis=1)).flatten()
        L = diags(degrees_lcc) - W
    else:
        lcc_idx = None

    return L, W, lcc_idx


def compute_spectrum(L: csr_matrix, n_modes: int = 50) -> tuple:
    """Compute first n_modes eigenvalues and eigenvectors."""
    n_modes = min(n_modes, L.shape[0] - 2)
    try:
        eigenvalues, eigenvectors = eigsh(L, k=n_modes, which="SM",
                                          maxiter=5000, tol=1e-8)
    except Exception:
        try:
            eigenvalues, eigenvectors = eigsh(L, k=n_modes, sigma=-0.01, which="LM")
        except Exception:
            L_dense = L.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            eigenvalues = eigenvalues[:n_modes]
            eigenvectors = eigenvectors[:, :n_modes]

    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def retrodict_mode_amplitudes(
    eigenvalues: np.ndarray,
    current_amplitudes: np.ndarray,
    delta_t_gyr: float,
    max_amplification: float = 1e4,
    tikhonov_alpha: float = 1e-6,
) -> dict:
    """
    Retrodict mode amplitudes at earlier cosmic time via inverse heat kernel.

    Uses spectral time: tau = 3 * H0^2 * Delta_t (NOT sqrt(3) * H0 * Delta_t).

    The heat kernel: c_k(t) = c_k(0) * exp(-lambda_k * tau)
    Retrodiction: c_k(0) = c_k(t) * exp(+lambda_k * tau) with Tikhonov regularization.
    """
    # Spectral time exponent: lambda_k * tau_s = (lambda_k/lambda_1) * 3*H0^2 * Delta_t
    delta_t_spectral = LAMBDA1_PHYS * delta_t_gyr  # dimensionless

    lam1 = eigenvalues[1] if len(eigenvalues) > 1 and eigenvalues[1] > 0 else 1.0
    lam_normalized = eigenvalues / lam1

    # Amplification exponents (on |c_k|^2)
    exponents = 2 * lam_normalized * delta_t_spectral
    log_max_amp = np.log(max_amplification)

    recoverable = exponents < log_max_amp
    n_recoverable = int(np.sum(recoverable))

    amplification = np.where(exponents < 500, np.exp(exponents), np.inf)

    # Tikhonov-regularized retrodiction
    reg_amp = np.zeros_like(amplification)
    reg_amp[recoverable] = amplification[recoverable] / (
        1 + tikhonov_alpha * amplification[recoverable]**2
    )
    early_amplitudes = current_amplitudes * reg_amp

    return {
        "early_amplitudes": early_amplitudes,
        "recoverable_mask": recoverable,
        "n_recoverable": n_recoverable,
        "delta_t_spectral": float(delta_t_spectral),
        "lam_normalized": lam_normalized,
    }


def identify_spectral_fixed_points(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    L: csr_matrix,
    n_modes: int = 10,
) -> dict:
    """
    Find spectral fixed points (local maxima of spectral centrality).

    Spectral centrality: C_i = sum_{k=1}^{n_modes} |v_k(i)|^2 / lambda_k
    Fixed points = predicted black hole formation sites.
    """
    N = L.shape[0]
    centrality = np.zeros(N)
    for k in range(1, min(n_modes + 1, len(eigenvalues))):
        if eigenvalues[k] > 1e-10:
            centrality += np.abs(eigenvectors[:, k])**2 / eigenvalues[k]

    # Extract adjacency for neighbor lookup
    W = -L.copy()
    W.setdiag(0)
    W.eliminate_zeros()

    is_local_max = np.zeros(N, dtype=bool)
    for i in range(N):
        neighbors = W[i].nonzero()[1]
        if len(neighbors) > 0:
            is_local_max[i] = centrality[i] > np.max(centrality[neighbors])

    fp_idx = np.where(is_local_max)[0]
    fp_centralities = centrality[fp_idx]
    order = np.argsort(fp_centralities)[::-1]

    return {
        "n_fixed_points": len(fp_idx),
        "centralities": fp_centralities[order].tolist(),
    }


def run_early_universe_analysis(
    positions: np.ndarray,
    densities: np.ndarray,
    linking_length: float = 25.0,
    n_modes: int = 50,
) -> dict:
    """
    Run the full spectral early universe analysis pipeline.

    Parameters
    ----------
    positions : (N, 3) array
    densities : (N,) array
    linking_length : float
    n_modes : int

    Returns
    -------
    dict with eigenvalue analysis, retrodiction results, and fixed points.
    """
    # Construct Laplacian
    L, W, lcc_idx = construct_cosmic_laplacian(positions, linking_length)

    if lcc_idx is not None:
        densities = densities[lcc_idx]

    n_modes = min(n_modes, L.shape[0] - 2)
    eigenvalues, eigenvectors = compute_spectrum(L, n_modes)

    # Project density onto eigenvector basis
    current_amplitudes = np.zeros(n_modes)
    for k in range(n_modes):
        current_amplitudes[k] = np.dot(eigenvectors[:, k], densities)**2

    total_energy = np.sum(current_amplitudes)
    if total_energy > 0:
        current_amplitudes /= total_energy

    ground_state_pct = float(current_amplitudes[0] * 100)

    # Retrodiction at z=6, 10, 14
    retro_results = {}
    for z_target, dt_gyr in LOOKBACK_TIMES.items():
        retro = retrodict_mode_amplitudes(
            eigenvalues, current_amplitudes, dt_gyr,
            max_amplification=1e8, tikhonov_alpha=1e-6,
        )
        retro_results[z_target] = {
            "n_recoverable": retro["n_recoverable"],
            "delta_t_spectral": retro["delta_t_spectral"],
        }

    # Fixed points
    fp_data = identify_spectral_fixed_points(eigenvectors, eigenvalues, L)

    lam1 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    lam2 = float(eigenvalues[2]) if len(eigenvalues) > 2 else 0.0
    lam2_lam1_ratio = lam2 / lam1 if lam1 > 0 else float("inf")

    return {
        "lambda_1": lam1,
        "lambda_2": lam2,
        "lambda2_lambda1_ratio": lam2_lam1_ratio,
        "ground_state_pct": ground_state_pct,
        "n_modes_computed": n_modes,
        "n_nodes": L.shape[0],
        "retrodiction": retro_results,
        "fixed_points": fp_data,
        "eigenvalues_first_10": eigenvalues[:10].tolist(),
    }


def main(filepath: Optional[str] = None):
    """Run the spectral early universe analysis."""
    print("Spectral Early Universe Analysis")
    print("=" * 60)

    if filepath is None:
        filepath = str(DEFAULT_CF4_PATH)

    if not Path(filepath).exists():
        print(f"CF4++ data not found at {filepath}")
        print("See cosmology/data/README.md for download instructions.")
        sys.exit(1)

    print("Loading CF4++ data...")
    positions, velocities, densities = load_cf4_grid(
        filepath, delta_threshold=0.0, max_points=10000,
    )
    print(f"  {len(positions)} cells loaded")

    print("\nRunning analysis...")
    result = run_early_universe_analysis(positions, densities)

    print(f"\nResults:")
    print(f"  lambda_1 = {result['lambda_1']:.6f}")
    print(f"  lambda_2 / lambda_1 = {result['lambda2_lambda1_ratio']:.1f}")
    print(f"  Ground state = {result['ground_state_pct']:.1f}%")

    for z, r in result["retrodiction"].items():
        print(f"  z={z}: {r['n_recoverable']} recoverable modes")

    print(f"  Spectral fixed points: {result['fixed_points']['n_fixed_points']}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    outpath = results_dir / "early_universe.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    main(filepath)
