#!/usr/bin/env python3
"""
Density-lambda1 correlation analysis.

Demonstrates that denser regions have higher spectral gap:
    Corr(rho, lambda_1) = +0.950

This is partially tautological (higher density -> more edges -> higher
lambda_1), but it validates the graph construction and motivates the
non-tautological test which controls for density.

Book reference: Chapter 37 -- Cosmology
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .data_loader import (
    DEFAULT_CF4_PATH,
    create_synthetic_subgrid_fields,
    load_cf4_subgrid_fields,
)
from .cf4_spectral_test import analyze_subregions, compute_correlations


def run_analysis(
    delta: np.ndarray,
    v_rad: np.ndarray,
    subgrid_size: int = 16,
    label: str = "data",
) -> dict:
    """
    Run the density-lambda1 correlation analysis.

    Parameters
    ----------
    delta : 3D density field
    v_rad : 3D radial velocity field
    subgrid_size : int
    label : str

    Returns
    -------
    dict with correlation results and environment statistics.
    """
    results = analyze_subregions(delta, v_rad, subgrid_size=subgrid_size)
    correlations = compute_correlations(results)

    deltas = results["deltas"]
    lambda1s = results["lambda1s"]

    # Environment analysis
    void_mask = deltas < -0.5
    filament_mask = (deltas >= -0.5) & (deltas <= 0.5)
    cluster_mask = deltas > 0.5

    env_stats = {}
    for name, mask in [("void", void_mask), ("filament", filament_mask), ("cluster", cluster_mask)]:
        if mask.sum() > 0:
            env_stats[name] = {
                "count": int(mask.sum()),
                "mean_lambda1": float(lambda1s[mask].mean()),
                "mean_delta": float(deltas[mask].mean()),
            }

    if void_mask.sum() > 0 and cluster_mask.sum() > 0:
        env_stats["cluster_void_ratio"] = float(
            lambda1s[cluster_mask].mean() / lambda1s[void_mask].mean()
        )

    return {
        "label": label,
        "correlations": correlations,
        "environment": env_stats,
    }


def main(filepath: Optional[str] = None):
    """Run the density-lambda1 correlation analysis."""
    print("Density-Lambda1 Correlation Analysis")
    print("=" * 60)

    use_synthetic = False

    if filepath is None:
        filepath = str(DEFAULT_CF4_PATH)

    if not Path(filepath).exists():
        print(f"CF4++ data not found at {filepath}")
        print("Running with synthetic data for validation.\n")
        use_synthetic = True

    if use_synthetic:
        delta, v_rad = create_synthetic_subgrid_fields(n_grid=64)
        label = "synthetic"
    else:
        delta, v_rad = load_cf4_subgrid_fields(filepath)
        label = "CF4++"

    print(f"Data: {label}, grid shape: {delta.shape}")

    result = run_analysis(delta, v_rad, label=label)
    corr = result["correlations"]

    print(f"\nResults ({corr['n_subregions']} subregions):")
    print(f"  Corr(rho, lambda_1) = {corr['r_lambda_delta']:+.3f} "
          f"(p = {corr['p_lambda_delta']:.2e})")
    print(f"  Spearman rho        = {corr['rho_lambda_delta']:+.3f}")

    if "environment" in result:
        env = result["environment"]
        for name in ["void", "filament", "cluster"]:
            if name in env:
                e = env[name]
                print(f"  {name.capitalize()}: n={e['count']}, "
                      f"mean lambda_1={e['mean_lambda1']:.4f}")
        if "cluster_void_ratio" in env:
            print(f"  Cluster/Void ratio: {env['cluster_void_ratio']:.2f}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    outpath = results_dir / "density_correlation.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    main(filepath)
