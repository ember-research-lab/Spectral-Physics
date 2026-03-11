"""Tests for the parameter census (Chapter 38).

Tests each derivation module, regression-tests against census.json,
and validates tier classification and experimental accuracy bounds.
"""

import json
import math
import os

import numpy as np
import pytest

# ── Import all modules under test ────────────────────────────────────────────

from parameter_census.constants import (
    phi, tau, lam, delta_cp,
    EXPERIMENT, TIER_1_LABELS, TIER_2_LABELS, TIER_3_LABELS,
)
from parameter_census.fine_structure import derive_alpha
from parameter_census.koide_formula import derive_koide, verify_koide_from_masses
from parameter_census.weinberg_angle import derive_weinberg_angle
from parameter_census.cosmological_constant import derive_cosmological_ratio
from parameter_census.baryon_asymmetry import derive_baryon_asymmetry
from parameter_census.coincidence_problem import derive_omega_DM, derive_omega_Lambda
from parameter_census.tier1_structural import (
    get_all_tier1, derive_theta_13, derive_theta_12, derive_theta_23,
)
from parameter_census.tier2_tau_postulate import get_all_tier2
from parameter_census.tier3_fitted import get_all_tier3
from parameter_census.summary_table import collect_all


# ── Regression reference ─────────────────────────────────────────────────────

CENSUS_JSON = os.path.join(
    os.path.dirname(__file__), os.pardir, "results", "census.json"
)


def _load_reference() -> list[dict]:
    """Load the reference census.json (must exist)."""
    with open(CENSUS_JSON) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Framework constants
# ═══════════════════════════════════════════════════════════════════════════════


class TestFrameworkConstants:
    """Verify the four framework parameters."""

    def test_golden_ratio(self):
        assert abs(phi - 1.6180339887498949) < 1e-14

    def test_tau(self):
        expected = 1 / (2 + phi)
        assert abs(tau - expected) < 1e-14
        assert abs(tau - 0.27639320225002106) < 1e-14

    def test_lambda(self):
        assert lam == 0.2240

    def test_delta_cp(self):
        assert delta_cp == 68.75

    def test_tau_identity(self):
        """tau satisfies 1/tau = 2 + phi."""
        assert abs(1 / tau - (2 + phi)) < 1e-14


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Individual derivations
# ═══════════════════════════════════════════════════════════════════════════════


class TestFineStructure:
    def test_formula(self):
        """alpha = tau^2 / (4 * phi^2)."""
        result = derive_alpha()
        expected = tau**2 / (4 * phi**2)
        assert abs(result["predicted"] - expected) < 1e-15

    def test_within_1_percent(self):
        result = derive_alpha()
        assert result["error_pct"] < 1.0, (
            f"alpha error {result['error_pct']:.3f}% exceeds 1%"
        )

    def test_tier(self):
        assert derive_alpha()["tier"] == 1

    def test_inverse_alpha_near_137(self):
        result = derive_alpha()
        inv = 1 / result["predicted"]
        assert 136 < inv < 138


class TestKoide:
    def test_formula(self):
        """K = 2/3 from circulant structure."""
        result = derive_koide()
        assert abs(result["predicted"] - 2 / 3) < 1e-15

    def test_within_0_1_percent(self):
        result = derive_koide()
        assert result["error_pct"] < 0.1

    def test_tier(self):
        assert derive_koide()["tier"] == 1

    def test_from_masses(self):
        """Cross-check: K computed from PDG masses matches."""
        check = verify_koide_from_masses()
        assert abs(check["K_from_masses"] - 2 / 3) < 0.001


class TestWeinbergAngle:
    def test_formula(self):
        result = derive_weinberg_angle()
        assert result["predicted"] == lam

    def test_within_5_percent(self):
        result = derive_weinberg_angle()
        assert result["error_pct"] < 5.0

    def test_tier(self):
        assert derive_weinberg_angle()["tier"] == 1


class TestCosmologicalConstant:
    def test_formula(self):
        result = derive_cosmological_ratio()
        assert result["predicted"] == 2.0

    def test_within_5_percent(self):
        result = derive_cosmological_ratio()
        assert result["error_pct"] < 5.0

    def test_tier(self):
        assert derive_cosmological_ratio()["tier"] == 1


class TestBaryonAsymmetry:
    def test_formula(self):
        result = derive_baryon_asymmetry()
        expected = lam**14
        assert abs(result["predicted"] - expected) < 1e-20

    def test_order_of_magnitude(self):
        """Baryon asymmetry should be within 1 order of magnitude."""
        result = derive_baryon_asymmetry()
        ratio = result["predicted"] / result["experimental"]
        assert 0.1 < ratio < 10.0

    def test_tier(self):
        assert derive_baryon_asymmetry()["tier"] == 1


class TestCoincidenceProblem:
    def test_omega_DM_formula(self):
        result = derive_omega_DM()
        assert abs(result["predicted"] - tau) < 1e-15

    def test_omega_Lambda_formula(self):
        result = derive_omega_Lambda()
        assert abs(result["predicted"] - (1 - tau)) < 1e-15

    def test_sum_to_one(self):
        dm = derive_omega_DM()["predicted"]
        de = derive_omega_Lambda()["predicted"]
        assert abs(dm + de - 1.0) < 1e-15

    def test_omega_DM_within_5_percent(self):
        assert derive_omega_DM()["error_pct"] < 5.0

    def test_tiers(self):
        assert derive_omega_DM()["tier"] == 1
        assert derive_omega_Lambda()["tier"] == 1


class TestPMNSAngles:
    def test_theta_13_formula(self):
        result = derive_theta_13()
        expected_deg = np.degrees(np.arcsin(lam / np.sqrt(2)))
        assert abs(result["predicted"] - expected_deg) < 1e-10

    def test_theta_12_formula(self):
        result = derive_theta_12()
        expected_deg = np.degrees(np.arctan(1 / phi))
        assert abs(result["predicted"] - expected_deg) < 1e-10

    def test_theta_23_formula(self):
        result = derive_theta_23()
        expected_deg = np.degrees(np.pi / 4 - lam**2 / 2)
        assert abs(result["predicted"] - expected_deg) < 1e-10

    def test_theta_13_within_10_percent(self):
        assert derive_theta_13()["error_pct"] < 10.0

    def test_theta_12_within_10_percent(self):
        assert derive_theta_12()["error_pct"] < 10.0

    def test_theta_23_within_10_percent(self):
        assert derive_theta_23()["error_pct"] < 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Tier structure
# ═══════════════════════════════════════════════════════════════════════════════


class TestTierClassification:
    def test_tier1_count(self):
        assert len(get_all_tier1()) == 10

    def test_tier2_count(self):
        assert len(get_all_tier2()) == 9

    def test_tier3_count(self):
        assert len(get_all_tier3()) == 1

    def test_total_count(self):
        assert len(collect_all()) == 20

    def test_all_tier1_marked(self):
        for r in get_all_tier1():
            assert r["tier"] == 1, f"{r['quantity']} should be tier 1"

    def test_all_tier2_marked(self):
        for r in get_all_tier2():
            assert r["tier"] == 2, f"{r['quantity']} should be tier 2"

    def test_all_tier3_marked(self):
        for r in get_all_tier3():
            assert r["tier"] == 3, f"{r['quantity']} should be tier 3"

    def test_tier_labels_complete(self):
        """Tier labels in constants.py cover all expected entries."""
        assert len(TIER_1_LABELS) == 10
        assert len(TIER_2_LABELS) == 9
        assert len(TIER_3_LABELS) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Result structure
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultStructure:
    """Every result dict has the required keys."""

    REQUIRED_KEYS = {"quantity", "formula", "predicted", "experimental",
                     "error_pct", "tier", "book_section"}

    def test_all_results_have_required_keys(self):
        for r in collect_all():
            missing = self.REQUIRED_KEYS - set(r.keys())
            assert not missing, (
                f"{r.get('quantity', '?')} missing keys: {missing}"
            )

    def test_all_predicted_are_finite(self):
        for r in collect_all():
            assert math.isfinite(r["predicted"]), (
                f"{r['quantity']}: predicted = {r['predicted']}"
            )

    def test_all_experimental_are_finite(self):
        for r in collect_all():
            assert math.isfinite(r["experimental"]), (
                f"{r['quantity']}: experimental = {r['experimental']}"
            )

    def test_all_errors_non_negative(self):
        for r in collect_all():
            assert r["error_pct"] >= 0, (
                f"{r['quantity']}: error_pct = {r['error_pct']}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Regression against census.json
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegression:
    """Regression test: current output matches saved census.json."""

    @pytest.fixture(scope="class")
    def reference(self):
        return _load_reference()

    @pytest.fixture(scope="class")
    def current(self):
        return collect_all()

    def test_same_count(self, reference, current):
        assert len(current) == len(reference), (
            f"Current has {len(current)} entries, reference has {len(reference)}"
        )

    def test_same_quantities(self, reference, current):
        ref_names = [r["quantity"] for r in reference]
        cur_names = [r["quantity"] for r in current]
        assert cur_names == ref_names

    def test_predicted_values_match(self, reference, current):
        """Predicted values must match to machine precision."""
        for ref, cur in zip(reference, current):
            assert abs(cur["predicted"] - ref["predicted"]) < 1e-12, (
                f"{cur['quantity']}: predicted changed from "
                f"{ref['predicted']} to {cur['predicted']}"
            )

    def test_experimental_values_match(self, reference, current):
        """Experimental reference values must not drift."""
        for ref, cur in zip(reference, current):
            assert abs(cur["experimental"] - ref["experimental"]) < 1e-12, (
                f"{cur['quantity']}: experimental changed from "
                f"{ref['experimental']} to {cur['experimental']}"
            )

    def test_tiers_match(self, reference, current):
        for ref, cur in zip(reference, current):
            assert cur["tier"] == ref["tier"], (
                f"{cur['quantity']}: tier changed from {ref['tier']} to {cur['tier']}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Experimental values sanity
# ═══════════════════════════════════════════════════════════════════════════════


class TestExperimentalValues:
    """Verify experimental reference values are physically sensible."""

    def test_alpha_inverse_near_137(self):
        alpha = EXPERIMENT["alpha"]["value"]
        assert 137.03 < 1 / alpha < 137.04

    def test_koide_near_two_thirds(self):
        assert abs(EXPERIMENT["koide_K"]["value"] - 2 / 3) < 0.001

    def test_weinberg_angle_in_range(self):
        sw2 = EXPERIMENT["sin2_theta_W"]["value"]
        assert 0.22 < sw2 < 0.24

    def test_omega_DM_in_range(self):
        assert 0.2 < EXPERIMENT["omega_DM"]["value"] < 0.35

    def test_omega_Lambda_in_range(self):
        assert 0.6 < EXPERIMENT["omega_Lambda"]["value"] < 0.8

    def test_baryon_asymmetry_order(self):
        eta = EXPERIMENT["baryon_asymmetry"]["value"]
        assert 1e-11 < eta < 1e-8

    def test_theta_13_in_range(self):
        assert 7 < EXPERIMENT["theta_13_PMNS"]["value"] < 10

    def test_proton_lifetime_bound(self):
        assert EXPERIMENT["proton_lifetime_years"]["value"] > 1e33
