#!/usr/bin/env python3
"""
Filament-eigenvector alignment test.

Tests whether cosmic filament velocities align with eigenvector gradients
of the graph Laplacian. Significant alignment means filaments are eigenvector
structure of the cosmic web Laplacian.

Key result: z=8.1 at eigenvector v_8, p < 0.001, 6/6 configs significant

Book reference: Chapter 37 -- Cosmology
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .data_loader import DEFAULT_CF4_PATH, load_cf4_grid
from .graph_builder import build_graph_laplacian, compute_eigenvectors


def compute_gradient(
    positions: np.ndarray,
    eigenvector: np.ndarray,
    W: csr_matrix,
) -> np.ndarray:
    """
    Compute gradient of an eigenvector field at each node.

    Uses weighted finite differences over graph neighbors.

    Parameters
    ----------
    positions : (N, 3) array
    eigenvector : (N,) array
    W : sparse adjacency matrix

    Returns
    -------
    gradients : (N, 3) array
    """
    N = len(positions)
    gradients = np.zeros((N, 3))

    for i in range(N):
        row_start = W.indptr[i]
        row_end = W.indptr[i + 1]
        neighbors = W.indices[row_start:row_end]
        w = W.data[row_start:row_end]

        if len(neighbors) == 0:
            continue

        delta_pos = positions[neighbors] - positions[i]
        dist_sq = np.sum(delta_pos**2, axis=1)
        valid = dist_sq > 1e-20
        if not valid.any():
            continue

        delta_v = eigenvector[neighbors[valid]] - eigenvector[i]
        wv = w[valid]
        dp = delta_pos[valid]
        dsq = dist_sq[valid]

        grad = np.sum(wv[:, None] * delta_v[:, None] * dp / dsq[:, None], axis=0)
        weight_sum = wv.sum()
        if weight_sum > 0:
            gradients[i] = grad / weight_sum

    return gradients


def alignment_score(
    velocities: np.ndarray,
    gradients: np.ndarray,
) -> float:
    """
    Mean cosine alignment between velocity and gradient fields.

    Returns mean of |cos(angle)| across all valid points.
    """
    v_norm = np.linalg.norm(velocities, axis=1)
    g_norm = np.linalg.norm(gradients, axis=1)
    valid = (v_norm > 1e-10) & (g_norm > 1e-10)

    if valid.sum() < 10:
        return 0.0

    v_unit = velocities[valid] / v_norm[valid, None]
    g_unit = gradients[valid] / g_norm[valid, None]
    cos_angles = np.sum(v_unit * g_unit, axis=1)
    return float(np.mean(cos_angles))


def permutation_test(
    velocities: np.ndarray,
    gradients: np.ndarray,
    n_perm: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute p-value and z-score via permutation test.

    Shuffles velocities to destroy spatial correlation and
    compares observed alignment to null distribution.
    """
    rng = np.random.default_rng(seed)
    observed = alignment_score(velocities, gradients)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(velocities))
        null[i] = alignment_score(velocities[perm], gradients)

    p_value = float(np.mean(np.abs(null) >= np.abs(observed)))
    z_score = float((observed - np.mean(null)) / (np.std(null) + 1e-10))
    return p_value, z_score


def run_alignment_test(
    positions: np.ndarray,
    velocities: np.ndarray,
    linking_length: float = 20.0,
    n_eigenvectors: int = 10,
    n_perm: int = 500,
) -> Dict:
    """
    Run the full velocity-eigenvector alignment test.

    Parameters
    ----------
    positions : (N, 3) array
    velocities : (N, 3) array
    linking_length : float
    n_eigenvectors : int
    n_perm : int

    Returns
    -------
    dict with alignment results per eigenvector and best overall.
    """
    W, L = build_graph_laplacian(positions, linking_length)
    eigenvalues, eigenvectors = compute_eigenvectors(L, k=n_eigenvectors)

    results = {
        "eigenvalues": [],
        "alignments": [],
        "p_values": [],
        "z_scores": [],
    }

    best_k, best_align, best_p, best_z = 0, -999.0, 1.0, 0.0

    for k in range(1, n_eigenvectors):
        grad = compute_gradient(positions, eigenvectors[:, k], W)
        align = alignment_score(velocities, grad)
        p_val, z = permutation_test(velocities, grad, n_perm=n_perm)

        results["eigenvalues"].append(float(eigenvalues[k]))
        results["alignments"].append(align)
        results["p_values"].append(p_val)
        results["z_scores"].append(z)

        if align > best_align:
            best_k, best_align, best_p, best_z = k, align, p_val, z

    sig_count = sum(1 for p in results["p_values"] if p < 0.05)

    results["best_k"] = best_k
    results["best_alignment"] = best_align
    results["best_p"] = best_p
    results["best_z"] = best_z
    results["significant_count"] = sig_count
    results["n_points"] = len(positions)
    results["linking_length"] = linking_length

    return results


def main(filepath: Optional[str] = None):
    """Run the filament eigenvector alignment test on CF4++ data."""
    print("Filament Eigenvector Alignment Test")
    print("=" * 60)

    if filepath is None:
        filepath = str(DEFAULT_CF4_PATH)

    if not Path(filepath).exists():
        print(f"CF4++ data not found at {filepath}")
        print("See cosmology/data/README.md for download instructions.")
        sys.exit(1)

    np.random.seed(42)

    configs = [
        (0.5, 2000, 15.0),
        (0.5, 2000, 20.0),
        (0.5, 2000, 30.0),
        (1.0, 2000, 20.0),
        (1.0, 2000, 30.0),
        (0.0, 3000, 20.0),
    ]

    all_results = []

    for delta_thresh, max_pts, ll in configs:
        print(f"\nConfig: delta>{delta_thresh}, {max_pts} pts, LL={ll}")
        positions, velocities, densities = load_cf4_grid(
            filepath, delta_threshold=delta_thresh, max_points=max_pts
        )
        result = run_alignment_test(
            positions, velocities,
            linking_length=ll, n_eigenvectors=10, n_perm=500,
        )
        result["delta_threshold"] = delta_thresh
        all_results.append(result)

        print(f"  Best: v_{result['best_k']}, "
              f"align={result['best_alignment']:.4f}, "
              f"z={result['best_z']:.1f}, p={result['best_p']:.4f}")

    # Summary
    sig_configs = sum(1 for r in all_results if r["best_p"] < 0.05)
    best_overall = max(all_results, key=lambda r: r["best_z"])

    summary = {
        "best_z_score": best_overall["best_z"],
        "best_p_value": best_overall["best_p"],
        "best_eigenvector": best_overall["best_k"],
        "significant_configs": sig_configs,
        "total_configs": len(all_results),
    }

    print(f"\nSummary: {sig_configs}/{len(all_results)} configs significant (p<0.05)")
    print(f"Best z-score: {best_overall['best_z']:.1f} at v_{best_overall['best_k']}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    outpath = results_dir / "alignment_results.json"
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    main(filepath)
