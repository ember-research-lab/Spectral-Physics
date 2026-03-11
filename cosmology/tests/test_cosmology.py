#!/usr/bin/env python3
"""
Tests for the cosmology module.

All tests use synthetic data and do NOT require CF4++ data.
Run with: uv run pytest cosmology/tests/ -v
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr

# Module imports
from cosmology.data_loader import (
    create_synthetic_subgrid_fields,
    create_synthetic_universe,
)
from cosmology.graph_builder import (
    build_graph_laplacian,
    build_subgrid_laplacian,
    compute_eigenvectors,
    compute_fiedler_value,
)
from cosmology.eigenstate import (
    compute_eigenstate,
    compute_eigenstate_from_velocities,
    eigenstate_energy_levels,
)
from cosmology.lambda_spectral_floor import (
    compute_spectral_floor,
    spectral_floor_units,
)
from cosmology.attractor_dynamics import (
    generate_gaussian_field,
    evolve_zeldovich,
    compute_self_alignment_score,
    run_attractor_simulation,
)
from cosmology.spectral_early_universe import (
    retrodict_mode_amplitudes,
)
from cosmology.non_tautological_test import (
    partial_correlation,
    incremental_r_squared,
)
from cosmology.cf4_spectral_test import (
    analyze_subregions,
    compute_correlations,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def synthetic_universe():
    """Create a synthetic universe for testing."""
    positions, velocities, densities = create_synthetic_universe(
        n_grid=16, box_mpc=200.0, seed=42,
    )
    return positions, velocities, densities


@pytest.fixture
def synthetic_fields():
    """Create synthetic subgrid fields for testing."""
    delta, v_rad = create_synthetic_subgrid_fields(n_grid=32, seed=42)
    return delta, v_rad


@pytest.fixture
def small_laplacian(synthetic_universe):
    """Build a small Laplacian for testing."""
    positions, _, _ = synthetic_universe
    # Use larger linking length to ensure connectivity
    W, L = build_graph_laplacian(positions, linking_length=30.0)
    return W, L


# ============================================================
# Data Loader Tests
# ============================================================

class TestDataLoader:
    def test_synthetic_universe_shape(self, synthetic_universe):
        positions, velocities, densities = synthetic_universe
        n = 16**3
        assert positions.shape == (n, 3)
        assert velocities.shape == (n, 3)
        assert densities.shape == (n,)

    def test_synthetic_universe_position_range(self, synthetic_universe):
        positions, _, _ = synthetic_universe
        # Box is 200 Mpc, centered at origin
        assert positions.min() > -101
        assert positions.max() < 101

    def test_synthetic_universe_has_density_gradient(self, synthetic_universe):
        _, _, densities = synthetic_universe
        # Density should have meaningful variation
        assert densities.std() > 0.1

    def test_synthetic_fields_shape(self, synthetic_fields):
        delta, v_rad = synthetic_fields
        assert delta.shape == (32, 32, 32)
        assert v_rad.shape == (32, 32, 32)

    def test_synthetic_fields_anticorrelation(self, synthetic_fields):
        """Synthetic v_rad should be anticorrelated with density."""
        delta, v_rad = synthetic_fields
        r, _ = pearsonr(delta.ravel(), v_rad.ravel())
        assert r < -0.3, f"Expected strong negative correlation, got r={r:.3f}"

    def test_synthetic_reproducible(self):
        """Same seed should produce same data."""
        p1, _, d1 = create_synthetic_universe(n_grid=8, seed=123)
        p2, _, d2 = create_synthetic_universe(n_grid=8, seed=123)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(d1, d2)


# ============================================================
# Graph Builder Tests
# ============================================================

class TestGraphBuilder:
    def test_laplacian_shape(self, small_laplacian):
        _, L = small_laplacian
        n = 16**3
        assert L.shape == (n, n)

    def test_laplacian_symmetric(self, small_laplacian):
        _, L = small_laplacian
        diff = L - L.T
        assert diff.nnz == 0 or np.abs(diff.data).max() < 1e-10

    def test_laplacian_positive_semidefinite(self, small_laplacian):
        """L should be positive semi-definite: all eigenvalues >= 0."""
        _, L = small_laplacian
        # Compute a few smallest eigenvalues
        eigenvalues, _ = eigsh(L, k=6, which="SM", maxiter=5000)
        eigenvalues = np.sort(eigenvalues)
        # Allow small numerical error
        assert eigenvalues[0] > -1e-6, f"Smallest eigenvalue: {eigenvalues[0]}"

    def test_laplacian_zero_eigenvalue(self, small_laplacian):
        """Connected graph has exactly one zero eigenvalue."""
        _, L = small_laplacian
        eigenvalues, _ = eigsh(L, k=3, which="SM", maxiter=5000)
        eigenvalues = np.sort(np.abs(eigenvalues))
        assert eigenvalues[0] < 1e-6, f"Smallest eigenvalue should be ~0, got {eigenvalues[0]}"

    def test_fiedler_value_positive(self, small_laplacian):
        _, L = small_laplacian
        lam1 = compute_fiedler_value(L)
        assert lam1 > 0, f"Fiedler value should be positive, got {lam1}"

    def test_adjacency_symmetric(self, small_laplacian):
        W, _ = small_laplacian
        diff = W - W.T
        assert diff.nnz == 0 or np.abs(diff.data).max() < 1e-10

    def test_adjacency_nonnegative(self, small_laplacian):
        W, _ = small_laplacian
        assert W.data.min() >= 0, "Adjacency weights must be non-negative"

    def test_subgrid_laplacian(self):
        """Test subgrid Laplacian construction."""
        density = np.random.default_rng(42).normal(0, 1, (8, 8, 8))
        L = build_subgrid_laplacian(density)
        assert L.shape == (512, 512)

        # Check PSD
        eigenvalues, _ = eigsh(L, k=3, which="SM", maxiter=3000)
        assert np.sort(eigenvalues)[0] > -1e-6

    def test_eigenvectors_orthogonal(self, small_laplacian):
        _, L = small_laplacian
        eigenvalues, eigenvectors = compute_eigenvectors(L, k=5)
        # Check orthogonality
        gram = eigenvectors.T @ eigenvectors
        np.testing.assert_allclose(gram, np.eye(5), atol=1e-6)

    def test_no_edges_raises(self):
        """Very small linking length should raise ValueError."""
        positions = np.array([[0, 0, 0], [100, 100, 100]], dtype=float)
        with pytest.raises(ValueError, match="No edges found"):
            build_graph_laplacian(positions, linking_length=0.001)

    def test_binary_weights(self, synthetic_universe):
        positions, _, _ = synthetic_universe
        W, L = build_graph_laplacian(positions, linking_length=30.0, weight_fn="binary")
        # Binary weights should all be 1.0
        unique_weights = np.unique(W.data)
        assert len(unique_weights) == 1
        assert unique_weights[0] == 1.0


# ============================================================
# Eigenstate Tests
# ============================================================

class TestEigenstate:
    def test_ground_state_dominant(self):
        state = compute_eigenstate()
        assert state.ground_state_fraction > 0.999, (
            f"Ground state should be > 99.9%, got {state.ground_state_fraction * 100:.4f}%"
        )

    def test_ground_state_99_9996(self):
        """The specific prediction: 99.9996% ground state."""
        state = compute_eigenstate()
        assert state.ground_state_fraction > 0.999990
        assert state.ground_state_fraction < 1.0

    def test_dipole_from_bulk_flow(self):
        state = compute_eigenstate(v_bulk=630.0)
        expected_c1 = 630.0 / 299792.458
        assert abs(state.c_1 - expected_c1) < 1e-10

    def test_quadrupole_small(self):
        state = compute_eigenstate()
        assert state.c_2 < 1e-4

    def test_eigenstate_from_velocities(self):
        velocities = np.random.default_rng(42).normal(0, 200, (1000, 3))
        state = compute_eigenstate_from_velocities(velocities)
        # Random velocities should have small bulk flow
        assert state.v_bulk < 50  # km/s (random should partially cancel)
        assert state.ground_state_fraction > 0.99999

    def test_energy_levels_sum_to_one(self):
        levels, occupations = eigenstate_energy_levels(5)
        np.testing.assert_allclose(occupations.sum(), 1.0, atol=1e-10)

    def test_energy_levels_monotonically_decrease(self):
        levels, occupations = eigenstate_energy_levels(5)
        for i in range(len(occupations) - 1):
            assert occupations[i] >= occupations[i + 1]


# ============================================================
# Lambda Spectral Floor Tests
# ============================================================

class TestSpectralFloor:
    def test_lambda_over_h0sq(self):
        result = compute_spectral_floor()
        expected = 3.0 * 0.685  # = 2.055
        assert abs(result.lambda_over_h0sq - expected) < 1e-10

    def test_lambda1_over_h0sq(self):
        result = compute_spectral_floor()
        assert result.lambda1_over_h0sq == 2.0

    def test_agreement_within_3_percent(self):
        result = compute_spectral_floor()
        assert result.agreement_pct > 97.0, (
            f"Expected > 97% agreement, got {result.agreement_pct:.1f}%"
        )

    def test_spectral_floor_units(self):
        units = spectral_floor_units()
        assert units["h0_sq_gyr2"] > 0
        assert units["lambda_gyr2"] > 0
        assert units["lambda1_gyr2"] > 0
        assert units["tau_s_gyr"] > 5  # Should be several Gyr


# ============================================================
# Synthetic Universe Correlation Tests
# ============================================================

class TestSyntheticCorrelations:
    """Test that synthetic universe produces expected correlations."""

    def test_density_lambda1_positive(self, synthetic_fields):
        """Denser regions should have higher lambda_1."""
        delta, v_rad = synthetic_fields
        results = analyze_subregions(delta, v_rad, subgrid_size=8)

        if len(results["deltas"]) < 10:
            pytest.skip("Too few valid subregions")

        r, p = pearsonr(results["lambda1s"], results["deltas"])
        assert r > 0.3, (
            f"Expected positive density-lambda_1 correlation, got r={r:.3f}"
        )

    def test_lambda1_vr_negative(self, synthetic_fields):
        """Higher lambda_1 should correspond to lower radial velocity."""
        delta, v_rad = synthetic_fields
        results = analyze_subregions(delta, v_rad, subgrid_size=8)

        if len(results["deltas"]) < 10:
            pytest.skip("Too few valid subregions")

        r, p = pearsonr(results["lambda1s"], results["v_rads"])
        assert r < -0.2, (
            f"Expected negative lambda_1-v_r correlation, got r={r:.3f}"
        )


# ============================================================
# Attractor Dynamics Tests
# ============================================================

class TestAttractorDynamics:
    def test_zeldovich_evolves(self):
        """Zeldovich approximation should produce increasing structure."""
        Phi = generate_gaussian_field(n_grid=32, box_mpc=200.0, seed=42)
        early = evolve_zeldovich(Phi, growth_factor=0.1, box_mpc=200.0)
        late = evolve_zeldovich(Phi, growth_factor=1.0, box_mpc=200.0)
        assert np.std(late) > np.std(early), (
            "Late-time field should have more structure than early"
        )

    def test_s_cosmic_increases(self):
        """S_cosmic should increase from z=10 to z=0."""
        results = run_attractor_simulation(n_grid=32, box_mpc=200.0, seed=42)
        s_values = [r.s_cosmic for r in results]
        # At minimum, z=0 should be greater than z=10
        assert s_values[-1] > s_values[0], (
            f"S_cosmic should increase: z=10={s_values[0]:.4f}, z=0={s_values[-1]:.4f}"
        )

    def test_self_alignment_score_bounded(self):
        density = np.random.default_rng(42).normal(0, 1, (8, 8, 8))
        S = compute_self_alignment_score(density, lambda1=0.5, stability=2.0)
        assert 0 <= S <= 1, f"S_cosmic should be in [0,1], got {S}"

    def test_self_alignment_nan_handling(self):
        density = np.random.default_rng(42).normal(0, 1, (8, 8, 8))
        S = compute_self_alignment_score(density, lambda1=np.nan, stability=1.0)
        assert np.isnan(S)


# ============================================================
# Early Universe / Retrodiction Tests
# ============================================================

class TestRetrodiction:
    def test_recoverable_modes_decrease_with_lookback(self):
        """More distant lookback should have fewer recoverable modes."""
        eigenvalues = np.array([0.0, 0.001, 0.005, 0.01, 0.05, 0.1])
        amplitudes = np.array([0.9, 0.05, 0.02, 0.01, 0.01, 0.01])

        r6 = retrodict_mode_amplitudes(eigenvalues, amplitudes, 12.80)
        r10 = retrodict_mode_amplitudes(eigenvalues, amplitudes, 13.18)
        r14 = retrodict_mode_amplitudes(eigenvalues, amplitudes, 13.40)

        assert r6["n_recoverable"] >= r10["n_recoverable"]
        assert r10["n_recoverable"] >= r14["n_recoverable"]

    def test_ground_state_always_recoverable(self):
        """Mode 0 (lambda=0) should always be recoverable."""
        eigenvalues = np.array([0.0, 0.001, 0.01, 0.1])
        amplitudes = np.array([0.9, 0.05, 0.03, 0.02])

        result = retrodict_mode_amplitudes(eigenvalues, amplitudes, 13.0)
        assert result["recoverable_mask"][0] is True or result["recoverable_mask"][0] == True

    def test_spectral_time_convention(self):
        """Verify tau_s = 3*H0^2*Delta_t, NOT sqrt(3)*H0*Delta_t."""
        eigenvalues = np.array([0.0, 1.0])
        amplitudes = np.array([0.9, 0.1])
        dt_gyr = 10.0

        result = retrodict_mode_amplitudes(eigenvalues, amplitudes, dt_gyr)

        H0_gyr_inv = 67.4 / 978.0
        expected_scale = 3 * H0_gyr_inv**2 * dt_gyr
        assert abs(result["delta_t_spectral"] - expected_scale) < 1e-6, (
            f"Expected tau_s = 3*H0^2*dt = {expected_scale:.6f}, "
            f"got {result['delta_t_spectral']:.6f}"
        )


# ============================================================
# Non-Tautological Test Components
# ============================================================

class TestNonTautological:
    def test_partial_correlation_removes_confound(self):
        """Partial correlation should reduce spurious correlation."""
        rng = np.random.default_rng(42)
        z = rng.normal(0, 1, 200)
        x = z + rng.normal(0, 0.5, 200)
        y = z + rng.normal(0, 0.5, 200)

        r_simple = pearsonr(x, y)[0]
        r_partial = partial_correlation(x, y, z)

        # Simple should be high (confounded by z)
        assert r_simple > 0.5
        # Partial should be lower (z removed)
        assert abs(r_partial) < abs(r_simple)

    def test_incremental_r2_nonnegative(self):
        """Adding a predictor cannot decrease R^2."""
        rng = np.random.default_rng(42)
        deltas = rng.normal(0, 1, 100)
        lambda1s = deltas + rng.normal(0, 0.5, 100)  # Correlated with density
        v_rads = -deltas + rng.normal(0, 1, 100)

        result = incremental_r_squared(deltas, lambda1s, v_rads)
        assert result["delta_r2"] >= -1e-10, (
            f"Delta R^2 should be non-negative, got {result['delta_r2']}"
        )


# ============================================================
# Regression Tests Against Results JSON
# ============================================================

class TestResultsRegression:
    """Check that saved results JSON files contain expected values."""

    def test_alignment_results(self):
        path = RESULTS_DIR / "alignment_results.json"
        with open(path) as f:
            data = json.load(f)
        assert data["best_z_score"] > 5.0
        assert data["best_p_value"] < 0.01
        assert data["significant_configs"] == 6

    def test_density_correlation(self):
        path = RESULTS_DIR / "density_correlation.json"
        with open(path) as f:
            data = json.load(f)
        assert data["r_lambda_delta"] > 0.9
        assert data["r_lambda_delta"] < 1.0

    def test_hubble_correlation(self):
        path = RESULTS_DIR / "hubble_correlation.json"
        with open(path) as f:
            data = json.load(f)
        assert data["r_pearson"] < 0  # Anticorrelation
        assert data["p_pearson"] < 0.05  # Significant

    def test_non_tautological(self):
        path = RESULTS_DIR / "non_tautological.json"
        with open(path) as f:
            data = json.load(f)
        assert data["partial_correlation"] > 0.1
        assert data["delta_r2"] > 0.01

    def test_attractor(self):
        path = RESULTS_DIR / "attractor.json"
        with open(path) as f:
            data = json.load(f)
        assert data["is_increasing"] is True
        assert data["pct_change"] > 40

    def test_early_universe(self):
        path = RESULTS_DIR / "early_universe.json"
        with open(path) as f:
            data = json.load(f)
        assert data["ground_state_pct"] > 95
        assert data["retrodiction"]["z10"]["n_recoverable"] == 3
        assert data["n_fixed_points"] == 13
