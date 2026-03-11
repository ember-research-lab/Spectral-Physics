"""Weinberg angle from the Cabibbo parameter.

Book reference: Chapter 38, Section 38.4 -- The Weak Mixing Angle

Derivation
----------
The weak mixing angle theta_W parametrizes the electroweak symmetry
breaking.  In the spectral framework, the SU(2)xU(1) gauge sector
arises from the Laplacian's block decomposition, and the mixing angle
is set by the Cabibbo parameter:

    sin^2(theta_W) = lambda

where lambda = 0.2240 is the Wolfenstein parametrization of the CKM
matrix.  The physical reason: the same spectral projection that
determines quark-generation mixing also fixes the gauge boson mixing.

Experimental: sin^2(theta_W) = 0.23122 +/- 0.00003 (MS-bar at M_Z).
Agreement: 3%.
"""

from .constants import lam, EXPERIMENT


def derive_weinberg_angle() -> dict:
    """Derive sin^2(theta_W) from the Cabibbo parameter.

    Returns
    -------
    dict with keys: predicted, experimental, error_pct, formula
    """
    predicted = lam
    experimental = EXPERIMENT["sin2_theta_W"]["value"]
    error_pct = abs(predicted - experimental) / experimental * 100

    return {
        "quantity": "sin^2(theta_W)",
        "formula": "lambda (Cabibbo parameter)",
        "predicted": predicted,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 1,
        "book_section": "22.4",
    }


if __name__ == "__main__":
    result = derive_weinberg_angle()
    print(f"Weak mixing angle")
    print(f"  Formula:      {result['formula']}")
    print(f"  Predicted:    sin^2(theta_W) = {result['predicted']:.4f}")
    print(f"  Experimental: sin^2(theta_W) = {result['experimental']:.5f}")
    print(f"  Error:        {result['error_pct']:.1f}%")
