#!/usr/bin/env python3
"""
Non-tautological test for spectral cosmology.

The lambda_1-density correlation (r=0.95) is partially tautological since
graph construction links density to connectivity. This script tests whether
lambda_1 predicts dynamics BEYOND what density alone predicts.

Key results:
    Partial correlation controlling for density: r = 0.174
    Incremental R^2: Delta_R^2 = 3.0%

Book reference: Chapter 37 -- Cosmology
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .data_loader import DEFAULT_CF4_PATH, load_cf4_subgrid_fields
from .cf4_spectral_test import analyze_subregions


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Partial correlation of x and y controlling for z.

    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))
    """
    r_xy = pearsonr(x, y)[0]
    r_xz = pearsonr(x, z)[0]
    r_yz = pearsonr(y, z)[0]

    numerator = r_xy - r_xz * r_yz
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    if denominator < 1e-10:
        return np.nan
    return numerator / denominator


def incremental_r_squared(
    deltas: np.ndarray,
    lambda1s: np.ndarray,
    v_rads: np.ndarray,
) -> dict:
    """
    Compute incremental R^2 from adding lambda_1 to a density-only model.

    Model 1: v_r ~ delta (density only)
    Model 2: v_r ~ delta + lambda_1 (density + spectral gap)

    Returns
    -------
    dict with r2_density, r2_both, delta_r2, and model coefficients.
    """
    X_density = deltas.reshape(-1, 1)
    model1 = LinearRegression().fit(X_density, v_rads)
    pred1 = model1.predict(X_density)
    r2_density = r2_score(v_rads, pred1)

    X_both = np.column_stack([deltas, lambda1s])
    model2 = LinearRegression().fit(X_both, v_rads)
    pred2 = model2.predict(X_both)
    r2_both = r2_score(v_rads, pred2)

    return {
        "r2_density": float(r2_density),
        "r2_both": float(r2_both),
        "delta_r2": float(r2_both - r2_density),
        "beta_delta_model1": float(model1.coef_[0]),
        "beta_delta_model2": float(model2.coef_[0]),
        "beta_lambda1_model2": float(model2.coef_[1]),
    }


def void_expansion_test(
    deltas: np.ndarray,
    lambda1s: np.ndarray,
    v_rads: np.ndarray,
) -> dict:
    """
    Test whether low-lambda_1 voids expand faster than density predicts.

    Residualize v_r against density, then check if lambda_1 predicts
    the residual in void regions.
    """
    # Residual velocity after removing density effect
    model = LinearRegression().fit(deltas.reshape(-1, 1), v_rads)
    v_r_pred = model.predict(deltas.reshape(-1, 1))
    v_r_residual = v_rads - v_r_pred

    # Focus on voids (lowest 25% density)
    void_mask = deltas < np.percentile(deltas, 25)
    n_voids = int(void_mask.sum())

    if n_voids < 20:
        return {"error": "Insufficient void regions", "n_voids": n_voids}

    r_void, p_void = pearsonr(lambda1s[void_mask], v_r_residual[void_mask])

    # Split voids by lambda_1
    void_lambda1s = lambda1s[void_mask]
    void_residuals = v_r_residual[void_mask]
    low_mask = void_lambda1s < np.median(void_lambda1s)

    mean_vr_low = float(void_residuals[low_mask].mean())
    mean_vr_high = float(void_residuals[~low_mask].mean())

    return {
        "n_voids": n_voids,
        "r_void": float(r_void),
        "p_void": float(p_void),
        "mean_vr_residual_low_lambda1": mean_vr_low,
        "mean_vr_residual_high_lambda1": mean_vr_high,
        "low_expands_faster": mean_vr_low > mean_vr_high,
    }


def run_non_tautological_test(
    delta_field: np.ndarray,
    vr_field: np.ndarray,
    subgrid_size: int = 16,
) -> dict:
    """
    Run the full non-tautological test suite.

    Parameters
    ----------
    delta_field : 3D density field
    vr_field : 3D radial velocity field
    subgrid_size : int

    Returns
    -------
    dict with partial correlation, incremental R^2, and void expansion results.
    """
    results = analyze_subregions(delta_field, vr_field, subgrid_size=subgrid_size)
    deltas = results["deltas"]
    v_rads = results["v_rads"]
    lambda1s = results["lambda1s"]

    # Test 1: Partial correlation
    r_partial = partial_correlation(lambda1s, v_rads, deltas)

    # Test 2: Incremental R^2
    incr = incremental_r_squared(deltas, lambda1s, v_rads)

    # Test 3: Void expansion
    void_result = void_expansion_test(deltas, lambda1s, v_rads)

    # Simple correlations for reference
    r_lambda_vr = float(pearsonr(lambda1s, v_rads)[0])
    r_lambda_delta = float(pearsonr(lambda1s, deltas)[0])
    r_delta_vr = float(pearsonr(deltas, v_rads)[0])

    return {
        "n_subregions": len(deltas),
        "simple_correlations": {
            "r_lambda_vr": r_lambda_vr,
            "r_lambda_delta": r_lambda_delta,
            "r_delta_vr": r_delta_vr,
        },
        "partial_correlation": float(r_partial),
        "incremental_r2": incr,
        "void_expansion": void_result,
    }


def main(filepath: Optional[str] = None):
    """Run the non-tautological test."""
    print("Non-Tautological Test for Spectral Cosmology")
    print("=" * 60)

    if filepath is None:
        filepath = str(DEFAULT_CF4_PATH)

    if not Path(filepath).exists():
        print(f"CF4++ data not found at {filepath}")
        print("See cosmology/data/README.md for download instructions.")
        sys.exit(1)

    print("Loading CF4++ data...")
    delta, v_rad = load_cf4_subgrid_fields(filepath)

    print("Running non-tautological tests...")
    result = run_non_tautological_test(delta, v_rad)

    print(f"\nResults ({result['n_subregions']} subregions):")

    sc = result["simple_correlations"]
    print(f"\n  Simple correlations:")
    print(f"    Corr(lambda_1, v_r)  = {sc['r_lambda_vr']:.4f}")
    print(f"    Corr(lambda_1, delta) = {sc['r_lambda_delta']:.4f}")
    print(f"    Corr(delta, v_r)      = {sc['r_delta_vr']:.4f}")

    print(f"\n  Test 1: Partial Correlation (controlling for density)")
    print(f"    Corr(lambda_1, v_r | delta) = {result['partial_correlation']:.4f}")

    incr = result["incremental_r2"]
    print(f"\n  Test 2: Incremental R^2")
    print(f"    R^2(delta only) = {incr['r2_density']:.4f}")
    print(f"    R^2(delta + lambda_1) = {incr['r2_both']:.4f}")
    print(f"    Delta R^2 = {incr['delta_r2']:.4f} ({incr['delta_r2'] * 100:.1f}%)")

    void = result["void_expansion"]
    if "error" not in void:
        print(f"\n  Test 3: Void Expansion")
        print(f"    n_voids = {void['n_voids']}")
        print(f"    Corr(lambda_1, v_r_residual) in voids = {void['r_void']:.4f}")
        print(f"    Low-lambda_1 expands faster: {void['low_expands_faster']}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    outpath = results_dir / "non_tautological.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    main(filepath)
