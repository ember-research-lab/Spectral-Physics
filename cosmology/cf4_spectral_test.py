#!/usr/bin/env python3
"""
Graph Laplacian from CF4++ galaxy positions.

Computes lambda_1 (Fiedler value) for 16^3 subregions of the CF4++
128^3 density grid. Measures correlations between density, spectral gap,
and radial velocity.

Key result: Corr(rho, lambda_1) = +0.950

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

# Analysis parameters
SUBGRID_SIZE = 16  # 16^3 subregions
SIGMA_WEIGHT = 2.0  # Gaussian weight scale for density-based edges


def analyze_subregions(
    delta: np.ndarray,
    v_rad: np.ndarray,
    subgrid_size: int = SUBGRID_SIZE,
) -> dict:
    """
    Divide 3D fields into subregions and compute spectral gap for each.

    Parameters
    ----------
    delta : (N, N, N) array
        Density contrast field.
    v_rad : (N, N, N) array
        Radial peculiar velocity field.
    subgrid_size : int
        Size of each subregion.

    Returns
    -------
    dict with keys: deltas, v_rads, lambda1s (arrays of per-subregion values)
    """
    n_grid = delta.shape[0]
    n_sub = n_grid // subgrid_size

    deltas, v_rads, lambda1s = [], [], []

    for i in range(n_sub):
        for j in range(n_sub):
            for k in range(n_sub):
                i0, i1 = i * subgrid_size, (i + 1) * subgrid_size
                j0, j1 = j * subgrid_size, (j + 1) * subgrid_size
                k0, k1 = k * subgrid_size, (k + 1) * subgrid_size

                sub_delta = delta[i0:i1, j0:j1, k0:k1]
                sub_vrad = v_rad[i0:i1, j0:j1, k0:k1]

                # Compute spectral gap for this subregion
                try:
                    L = build_subgrid_laplacian(sub_delta, sigma_weight=SIGMA_WEIGHT)
                    lam1 = compute_fiedler_value(L)
                except (ValueError, Exception):
                    lam1 = np.nan

                if not np.isnan(lam1):
                    deltas.append(np.mean(sub_delta))
                    v_rads.append(np.mean(sub_vrad))
                    lambda1s.append(lam1)

    return {
        "deltas": np.array(deltas),
        "v_rads": np.array(v_rads),
        "lambda1s": np.array(lambda1s),
    }


def compute_correlations(results: dict) -> dict:
    """
    Compute correlation statistics from subregion analysis.

    Returns
    -------
    dict with correlation coefficients and p-values.
    """
    deltas = results["deltas"]
    v_rads = results["v_rads"]
    lambda1s = results["lambda1s"]

    r_lambda_delta, p_lambda_delta = pearsonr(lambda1s, deltas)
    rho_lambda_delta, _ = spearmanr(lambda1s, deltas)

    r_lambda_vr, p_lambda_vr = pearsonr(lambda1s, v_rads)
    rho_lambda_vr, _ = spearmanr(lambda1s, v_rads)

    r_delta_vr, p_delta_vr = pearsonr(deltas, v_rads)

    return {
        "n_subregions": len(deltas),
        "r_lambda_delta": float(r_lambda_delta),
        "p_lambda_delta": float(p_lambda_delta),
        "rho_lambda_delta": float(rho_lambda_delta),
        "r_lambda_vr": float(r_lambda_vr),
        "p_lambda_vr": float(p_lambda_vr),
        "rho_lambda_vr": float(rho_lambda_vr),
        "r_delta_vr": float(r_delta_vr),
        "p_delta_vr": float(p_delta_vr),
    }


def main(filepath: Optional[str] = None):
    """Run the CF4++ spectral gap analysis."""
    print("CF4++ Spectral Gap Analysis")
    print("=" * 60)

    if filepath is None:
        filepath = str(DEFAULT_CF4_PATH)

    if not Path(filepath).exists():
        print(f"CF4++ data not found at {filepath}")
        print("See cosmology/data/README.md for download instructions.")
        sys.exit(1)

    print("Loading CF4++ data...")
    delta, v_rad = load_cf4_subgrid_fields(filepath)
    print(f"  Grid shape: {delta.shape}")
    print(f"  Density range: [{delta.min():.2f}, {delta.max():.2f}]")
    print(f"  Velocity range: [{v_rad.min():.0f}, {v_rad.max():.0f}] km/s")

    n_sub = delta.shape[0] // SUBGRID_SIZE
    print(f"\nAnalyzing {n_sub}^3 = {n_sub**3} subregions...")

    results = analyze_subregions(delta, v_rad)
    correlations = compute_correlations(results)

    print(f"\nValid subregions: {correlations['n_subregions']}")

    print(f"\nlambda_1 vs density:")
    print(f"  Pearson r = {correlations['r_lambda_delta']:.3f} "
          f"(p = {correlations['p_lambda_delta']:.2e})")
    print(f"  Spearman rho = {correlations['rho_lambda_delta']:.3f}")

    print(f"\nlambda_1 vs radial velocity:")
    print(f"  Pearson r = {correlations['r_lambda_vr']:.3f} "
          f"(p = {correlations['p_lambda_vr']:.2e})")

    print(f"\ndelta vs v_r (sanity check):")
    print(f"  Pearson r = {correlations['r_delta_vr']:.3f}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    outpath = results_dir / "density_correlation.json"
    with open(outpath, "w") as f:
        json.dump(correlations, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    main(filepath)
