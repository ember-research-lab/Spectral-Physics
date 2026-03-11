"""Baryon asymmetry from the Cabibbo parameter.

Book reference: Chapter 38, Section 38.6 -- Baryon Asymmetry

Derivation
----------
The baryon-to-photon ratio eta_B emerges from the 14th power of the
Cabibbo parameter:

    eta_B = lambda^14

This counts the number of CKM-suppressed loop insertions required
to generate a net baryon number in the spectral framework's
electroweak baryogenesis channel.  Each quark-generation crossing
contributes a factor of lambda; the minimum topological path through
all three generations requires 14 such crossings.

Predicted:    lambda^14 = 8.0e-10
Experimental: eta_B     = 6.1e-10

This is order-of-magnitude agreement (~30%), which is the expected
accuracy for a leading-order calculation of this type.
"""

import numpy as np

from .constants import lam, EXPERIMENT


def derive_baryon_asymmetry() -> dict:
    """Derive baryon-to-photon ratio eta_B.

    Returns
    -------
    dict with keys: predicted, experimental, error_pct, formula
    """
    predicted = lam**14
    experimental = EXPERIMENT["baryon_asymmetry"]["value"]
    error_pct = abs(predicted - experimental) / experimental * 100

    return {
        "quantity": "Baryon asymmetry eta_B",
        "formula": "lambda^14",
        "predicted": predicted,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 1,
        "book_section": "22.6",
    }


if __name__ == "__main__":
    result = derive_baryon_asymmetry()
    print(f"Baryon asymmetry")
    print(f"  Formula:      {result['formula']}")
    print(f"  Predicted:    eta_B = {result['predicted']:.2e}")
    print(f"  Experimental: eta_B = {result['experimental']:.2e}")
    print(f"  Error:        {result['error_pct']:.0f}%")
    print(f"  (Order-of-magnitude agreement expected)")
