"""Dark-matter and dark-energy densities from the stability tolerance.

Book reference: Chapter 38, Section 38.7 -- The Coincidence Problem

Derivation
----------
The "coincidence problem" asks why Omega_DM and Omega_Lambda are
comparable today.  In the spectral framework this is not a coincidence:
both are set by the stability tolerance tau = 1/(2+phi).

    Omega_DM     = tau         = 0.276
    Omega_Lambda = 1 - tau     = 0.724

The physical interpretation: tau is the fraction of spectral weight
below the stability gap.  Modes below the gap behave as pressureless
matter (DM); modes above behave as dark energy.

Experimental (Planck 2018):
    Omega_DM     = 0.265 +/- 0.007
    Omega_Lambda = 0.685 +/- 0.007

Agreement: 4% for Omega_DM, 6% for Omega_Lambda.

Note: the experimental values don't sum to 1 because Omega_baryon ~ 0.049
accounts for the remainder.  Our prediction Omega_DM + Omega_Lambda = 1
applies to the dark sector only.
"""

from .constants import tau, EXPERIMENT


def derive_omega_DM() -> dict:
    """Derive dark-matter density parameter.

    Returns
    -------
    dict with keys: predicted, experimental, error_pct, formula
    """
    predicted = tau
    experimental = EXPERIMENT["omega_DM"]["value"]
    error_pct = abs(predicted - experimental) / experimental * 100

    return {
        "quantity": "Omega_DM",
        "formula": "tau = 1/(2+phi)",
        "predicted": predicted,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 1,
        "book_section": "22.7",
    }


def derive_omega_Lambda() -> dict:
    """Derive dark-energy density parameter.

    Returns
    -------
    dict with keys: predicted, experimental, error_pct, formula
    """
    predicted = 1 - tau
    experimental = EXPERIMENT["omega_Lambda"]["value"]
    error_pct = abs(predicted - experimental) / experimental * 100

    return {
        "quantity": "Omega_Lambda",
        "formula": "1 - tau",
        "predicted": predicted,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 1,
        "book_section": "22.7",
    }


if __name__ == "__main__":
    dm = derive_omega_DM()
    de = derive_omega_Lambda()
    print(f"Coincidence problem resolution")
    print(f"  Omega_DM:     predicted = {dm['predicted']:.3f},  "
          f"experiment = {dm['experimental']:.3f},  "
          f"error = {dm['error_pct']:.1f}%")
    print(f"  Omega_Lambda: predicted = {de['predicted']:.3f},  "
          f"experiment = {de['experimental']:.3f},  "
          f"error = {de['error_pct']:.1f}%")
    print(f"  Sum (dark sector): {dm['predicted'] + de['predicted']:.3f}")
