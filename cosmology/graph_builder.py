#!/usr/bin/env python3
"""
Graph Laplacian construction from galaxy positions.

Builds a weighted graph from a 3D point cloud using cKDTree for
neighbor queries and Gaussian distance weights. Returns sparse
Laplacian L = D - W suitable for spectral analysis.

Book reference: Chapter 37 -- Cosmology
"""

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


def build_graph_laplacian(
    positions: np.ndarray,
    linking_length: float,
    sigma: Optional[float] = None,
    weight_fn: str = "gaussian",
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Build adjacency matrix and graph Laplacian from 3D positions.

    Uses cKDTree for efficient neighbor queries. Edges connect all
    pairs within linking_length, weighted by Gaussian kernel.

    Parameters
    ----------
    positions : (N, 3) array
        3D coordinates (e.g., galaxy positions in Mpc).
    linking_length : float
        Maximum distance for edge creation.
    sigma : float or None
        Gaussian kernel width. Defaults to linking_length / 3.
    weight_fn : str
        Weight function: "gaussian" or "binary".

    Returns
    -------
    W : (N, N) sparse matrix
        Symmetric adjacency (weight) matrix.
    L : (N, N) sparse matrix
        Graph Laplacian L = D - W.
    """
    N = len(positions)
    if sigma is None:
        sigma = linking_length / 3.0

    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=linking_length, output_type="ndarray")

    if len(pairs) == 0:
        raise ValueError(
            f"No edges found with linking_length={linking_length}. "
            "Try increasing linking_length or checking position range."
        )

    # Compute distances and weights
    dists = np.linalg.norm(
        positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1
    )

    if weight_fn == "gaussian":
        weights = np.exp(-dists**2 / (2 * sigma**2))
    elif weight_fn == "binary":
        weights = np.ones_like(dists)
    else:
        raise ValueError(f"Unknown weight_fn: {weight_fn}")

    # Build symmetric sparse adjacency matrix
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.concatenate([weights, weights])
    W = csr_matrix((data, (rows, cols)), shape=(N, N))

    # Laplacian L = D - W
    degrees = np.array(W.sum(axis=1)).flatten()
    L = diags(degrees) - W

    return W, L


def build_subgrid_laplacian(
    density_subgrid: np.ndarray,
    sigma_weight: float = 2.0,
) -> csr_matrix:
    """
    Build graph Laplacian for a 3D density subgrid using 6-connectivity.

    Edge weights depend on local density: higher mean density between
    neighbors gives stronger connections.

    Parameters
    ----------
    density_subgrid : (nx, ny, nz) array
        3D density contrast field.
    sigma_weight : float
        Scale for density-based edge weight: w = exp(mean_delta / sigma).

    Returns
    -------
    L : (N, N) sparse matrix
        Graph Laplacian where N = nx * ny * nz.
    """
    nx, ny, nz = density_subgrid.shape
    N = nx * ny * nz

    if N < 10:
        raise ValueError(f"Subgrid too small: {N} nodes (need >= 10)")

    rows, cols, weights = [], [], []
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                node = i * ny * nz + j * nz + k
                d_node = density_subgrid[i, j, k]

                for di, dj, dk in directions:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        neighbor = ni * ny * nz + nj * nz + nk
                        d_neighbor = density_subgrid[ni, nj, nk]
                        mean_d = (d_node + d_neighbor) / 2
                        w = np.exp(mean_d / sigma_weight)

                        rows.append(node)
                        cols.append(neighbor)
                        weights.append(w)

    W = csr_matrix((weights, (rows, cols)), shape=(N, N))
    degrees = np.array(W.sum(axis=1)).flatten()
    D_sparse = diags(degrees)
    L = D_sparse - W

    return L


def compute_fiedler_value(
    L: csr_matrix,
    k: int = 6,
) -> float:
    """
    Compute the Fiedler value (smallest non-zero eigenvalue) of a Laplacian.

    Parameters
    ----------
    L : sparse matrix
        Graph Laplacian (positive semi-definite).
    k : int
        Number of smallest eigenvalues to compute.

    Returns
    -------
    lambda_1 : float
        The Fiedler value (algebraic connectivity).
    """
    N = L.shape[0]
    k = min(k, N - 2)
    if k < 2:
        return np.nan

    try:
        eigenvalues, _ = eigsh(L, k=k, which="SM", maxiter=5000)
        eigenvalues = np.sort(np.abs(eigenvalues))
        # Return first eigenvalue clearly above zero
        for ev in eigenvalues[1:]:
            if ev > 1e-8:
                return float(ev)
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else np.nan
    except Exception:
        return np.nan


def compute_eigenvectors(
    L: csr_matrix,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k smallest eigenvectors of a Laplacian.

    Parameters
    ----------
    L : sparse matrix
        Graph Laplacian.
    k : int
        Number of eigenpairs to compute.

    Returns
    -------
    eigenvalues : (k,) array
        Sorted eigenvalues.
    eigenvectors : (N, k) array
        Corresponding eigenvectors as columns.
    """
    N = L.shape[0]
    k = min(k, N - 2)

    try:
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM", maxiter=5000, tol=1e-8)
    except Exception:
        try:
            eigenvalues, eigenvectors = eigsh(L, k=k, sigma=-0.01, which="LM")
        except Exception:
            L_dense = L.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]

    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


if __name__ == "__main__":
    from data_loader import create_synthetic_universe

    print("Graph Builder -- Cosmology Module")
    print("=" * 50)

    positions, velocities, densities = create_synthetic_universe(n_grid=16)
    print(f"Synthetic universe: {len(positions)} points")

    W, L = build_graph_laplacian(positions, linking_length=20.0)
    degrees = np.array(W.sum(axis=1)).flatten()
    print(f"Graph: {W.nnz // 2} edges, mean degree {degrees.mean():.1f}")

    lambda1 = compute_fiedler_value(L)
    print(f"Fiedler value lambda_1 = {lambda1:.6f}")
