#!/usr/bin/env python3
"""
Hubble parameter - lambda_1 anticorrelation test.

Tests the spectral cosmology prediction that regions with higher
spectral gap have lower local Hubble parameter:

    Corr(lambda_1, H_local) = -0.183 (p = 0.010)

This is a critical falsification test: if dark energy = spectral gap,
then H ~ 1/lambda_1 implies anticorrelation.

Book reference: Chapter 37 -- Cosmology
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .data_loader import DEFAULT_CF4_PATH, load_cf4_subgrid_fields
from .graph_builder import build_subgrid_laplacian, compute_fiedler_value

# Cosmological constants
H0 = 67.4  # km/s/Mpc (Planck 2018)
C_LIGHT = 299792.458  # km/s


def compute_local_hubble(
    v_rad: float,
    distance: float,
    h0: float = H0,
) -> float:
    """
    Compute local Hubble parameter from radial velocity and distance.

    H_local = H0 + v_pec / d

    Parameters
    ----------
    v_rad : float
        Radial peculiar velocity in km/s.
    distance : float
        Comoving distance in Mpc.
    h0 : float
        Background Hubble constant.

    Returns
    -------
    H_local in km/s/Mpc, or NaN if distance too small.
    """
    if distance < 1.0:
        return np.nan
    return h0 + v_rad / distance


def analyze_lambda1_hubble(
    delta_field: np.ndarray,
    vr_field: np.ndarray,
    patch_size: int = 10,
    n_samples: int = 200,
    seed: int = 42,
) -> dict:
    """
    Sample patches, compute lambda_1 and local H, measure correlation.

    Parameters
    ----------
    delta_field : (N, N, N) density contrast
    vr_field : (N, N, N) radial velocity
    patch_size : int
    n_samples : int
    seed : int

    Returns
    -------
    dict with arrays and correlation statistics.
    """
    rng = np.random.default_rng(seed)
    N = delta_field.shape[0]
    L_box = 1000.0  # Mpc
    cell_size = L_box / N
    margin = 20
    center = N // 2

    lambda1_list = []
    h_local_list = []
    delta_list = []
    distance_list = []

    attempts = 0
    max_attempts = n_samples * 5

    while len(lambda1_list) < n_samples and attempts < max_attempts:
        attempts += 1

        cx = rng.integers(margin, N - margin - patch_size)
        cy = rng.integers(margin, N - margin - patch_size)
        cz = rng.integers(margin, N - margin - patch_size)

        delta_patch = delta_field[cx:cx + patch_size, cy:cy + patch_size, cz:cz + patch_size]

        if np.any(np.isnan(delta_patch)) or np.any(np.abs(delta_patch) > 10):
            continue

        # Compute lambda_1
        try:
            L = build_subgrid_laplacian(delta_patch)
            lam1 = compute_fiedler_value(L)
        except (ValueError, Exception):
            continue

        if np.isnan(lam1) or lam1 < 1e-10:
            continue

        # Distance from observer
        sgx = (cx + patch_size / 2 - center) * cell_size
        sgy = (cy + patch_size / 2 - center) * cell_size
        sgz = (cz + patch_size / 2 - center) * cell_size
        d = np.sqrt(sgx**2 + sgy**2 + sgz**2)

        if d < 20 or d > 400:
            continue

        # Radial velocity at patch center
        cx_c = cx + patch_size // 2
        cy_c = cy + patch_size // 2
        cz_c = cz + patch_size // 2
        vr = vr_field[cx_c, cy_c, cz_c]

        if np.isnan(vr):
            continue

        h_local = compute_local_hubble(vr, d)
        if np.isnan(h_local) or h_local < 30 or h_local > 150:
            continue

        lambda1_list.append(lam1)
        h_local_list.append(h_local)
        delta_list.append(np.mean(delta_patch))
        distance_list.append(d)

    lambda1_arr = np.array(lambda1_list)
    h_arr = np.array(h_local_list)
    delta_arr = np.array(delta_list)

    if len(lambda1_arr) < 10:
        return {"error": "Insufficient samples", "n_samples": len(lambda1_arr)}

    r_pearson, p_pearson = pearsonr(lambda1_arr, h_arr)
    r_spearman, p_spearman = spearmanr(lambda1_arr, h_arr)
    r_delta, p_delta = pearsonr(lambda1_arr, delta_arr)

    return {
        "n_samples": len(lambda1_arr),
        "r_pearson": float(r_pearson),
        "p_pearson": float(p_pearson),
        "r_spearman": float(r_spearman),
        "p_spearman": float(p_spearman),
        "r_lambda_delta": float(r_delta),
        "p_lambda_delta": float(p_delta),
        "h_mean": float(np.mean(h_arr)),
        "h_std": float(np.std(h_arr)),
        "lambda1_mean": float(np.mean(lambda1_arr)),
        "lambda1_std": float(np.std(lambda1_arr)),
    }


def main(filepath: Optional[str] = None):
    """Run the lambda_1 vs Hubble parameter anticorrelation test."""
    print("Lambda_1 - Hubble Parameter Anticorrelation Test")
    print("=" * 60)

    if filepath is None:
        filepath = str(DEFAULT_CF4_PATH)

    if not Path(filepath).exists():
        print(f"CF4++ data not found at {filepath}")
        print("See cosmology/data/README.md for download instructions.")
        sys.exit(1)

    print("Loading CF4++ data...")
    delta, v_rad = load_cf4_subgrid_fields(filepath)

    print("Sampling patches and computing lambda_1 vs H_local...")
    result = analyze_lambda1_hubble(delta, v_rad)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"\nResults ({result['n_samples']} samples):")
    print(f"  Corr(lambda_1, H_local):")
    print(f"    Pearson r  = {result['r_pearson']:.3f} (p = {result['p_pearson']:.3e})")
    print(f"    Spearman r = {result['r_spearman']:.3f} (p = {result['p_spearman']:.3e})")
    print(f"  Prediction: r < 0 (higher lambda_1 -> lower H)")
    confirmed = result["r_pearson"] < 0 and result["p_pearson"] < 0.05
    print(f"  Result: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    outpath = results_dir / "hubble_correlation.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    main(filepath)
