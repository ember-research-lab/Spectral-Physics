"""Tier 1: Structural parameters -- no free inputs.

Book reference: Chapter 38, Section 38.8 -- Tier 1 Census

These 10 quantities are derived purely from the framework's structural
parameters (phi, tau, lambda).  They require no fitting and no external
data.  Every one is a falsifiable prediction: if experiment deviates
beyond the stated error, the framework is wrong.

Tier 1 parameters:
    1.  Fine-structure constant alpha
    2.  Koide ratio K
    3.  sin^2(theta_W) (Weinberg angle)
    4.  Lambda / H_0^2 (cosmological ratio)
    5.  Omega_DM (dark matter density)
    6.  Omega_Lambda (dark energy density)
    7.  Baryon asymmetry eta_B
    8.  PMNS theta_13
    9.  PMNS theta_12
    10. PMNS theta_23
"""

import numpy as np

from .constants import phi, tau, lam, EXPERIMENT
from .fine_structure import derive_alpha
from .koide_formula import derive_koide
from .weinberg_angle import derive_weinberg_angle
from .cosmological_constant import derive_cosmological_ratio
from .baryon_asymmetry import derive_baryon_asymmetry
from .coincidence_problem import derive_omega_DM, derive_omega_Lambda


def derive_theta_13() -> dict:
    """PMNS theta_13 = arcsin(lambda / sqrt(2)).

    Book reference: Section 38.8.1

    The reactor mixing angle arises from a single Cabibbo insertion
    in the off-diagonal Laplacian block connecting the first and
    third neutrino generations, divided by the geometric factor sqrt(2)
    from the SU(2) doublet structure.
    """
    predicted_rad = np.arcsin(lam / np.sqrt(2))
    predicted_deg = np.degrees(predicted_rad)
    experimental = EXPERIMENT["theta_13_PMNS"]["value"]
    error_pct = abs(predicted_deg - experimental) / experimental * 100

    return {
        "quantity": "PMNS theta_13",
        "formula": "arcsin(lambda / sqrt(2))",
        "predicted": predicted_deg,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 1,
        "book_section": "22.8.1",
    }


def derive_theta_12() -> dict:
    """PMNS theta_12 = arctan(1 / phi).

    Book reference: Section 38.8.2

    The solar mixing angle is set by the golden ratio: the two
    lightest neutrino mass eigenstates mix with a tangent equal
    to 1/phi, reflecting the golden-ratio spacing of the first
    two Laplacian eigenvalues in the neutrino sector.
    """
    predicted_rad = np.arctan(1 / phi)
    predicted_deg = np.degrees(predicted_rad)
    experimental = EXPERIMENT["theta_12_PMNS"]["value"]
    error_pct = abs(predicted_deg - experimental) / experimental * 100

    return {
        "quantity": "PMNS theta_12",
        "formula": "arctan(1/phi)",
        "predicted": predicted_deg,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 1,
        "book_section": "22.8.2",
    }


def derive_theta_23() -> dict:
    """PMNS theta_23 = pi/4 - lambda^2/2.

    Book reference: Section 38.8.3

    The atmospheric mixing angle is near-maximal (pi/4) with a
    correction of order lambda^2, arising from the second-order
    Cabibbo suppression in the mu-tau Laplacian block.
    """
    predicted_rad = np.pi / 4 - lam**2 / 2
    predicted_deg = np.degrees(predicted_rad)
    experimental = EXPERIMENT["theta_23_PMNS"]["value"]
    error_pct = abs(predicted_deg - experimental) / experimental * 100

    return {
        "quantity": "PMNS theta_23",
        "formula": "pi/4 - lambda^2/2",
        "predicted": predicted_deg,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 1,
        "book_section": "22.8.3",
    }


def get_all_tier1() -> list[dict]:
    """Return all 10 Tier 1 derivations."""
    return [
        derive_alpha(),
        derive_koide(),
        derive_weinberg_angle(),
        derive_cosmological_ratio(),
        derive_omega_DM(),
        derive_omega_Lambda(),
        derive_baryon_asymmetry(),
        derive_theta_13(),
        derive_theta_12(),
        derive_theta_23(),
    ]


if __name__ == "__main__":
    print("=" * 72)
    print("TIER 1: Structural parameters (no free inputs)")
    print("=" * 72)
    for r in get_all_tier1():
        print(f"\n  {r['quantity']}")
        print(f"    Formula:      {r['formula']}")
        print(f"    Predicted:    {r['predicted']:.7g}")
        print(f"    Experimental: {r['experimental']:.7g}")
        print(f"    Error:        {r['error_pct']:.2f}%")
