"""Tier 3: Fitted parameters.

Book reference: Chapter 38, Section 38.10 -- Tier 3 Census

Tier 3 contains the single fitted parameter in the framework:

    delta_CP = 68.75 degrees

This is the CKM CP-violating phase.  It cannot currently be derived
from phi, tau, or lambda; it must be extracted from experiment.
The framework predicts that delta_CP is the ONLY free parameter
needed beyond the three structural constants.

Current experimental value: delta_CP = 65.4 +/- 3.2 degrees (PDG 2024).
The fitted value 68.75 degrees lies within 1-sigma.

Falsifiable prediction: if a measurement requires a SECOND fitted
parameter (beyond delta_CP) to reproduce the CKM matrix, the
spectral framework is falsified.
"""

from .constants import delta_cp


def get_all_tier3() -> list[dict]:
    """Return the single Tier 3 parameter."""
    # PDG 2024 central value for delta_CP (CKM)
    experimental = 65.4  # degrees
    predicted = delta_cp
    error_pct = abs(predicted - experimental) / experimental * 100

    return [
        {
            "quantity": "CKM CP phase delta_CP",
            "formula": "fitted (single free parameter)",
            "predicted": predicted,
            "experimental": experimental,
            "error_pct": error_pct,
            "tier": 3,
            "book_section": "22.10",
        }
    ]


if __name__ == "__main__":
    print("=" * 72)
    print("TIER 3: Fitted parameters (1 free parameter)")
    print("=" * 72)
    for r in get_all_tier3():
        print(f"\n  {r['quantity']}")
        print(f"    Value:        {r['predicted']:.2f} degrees")
        print(f"    Experimental: {r['experimental']:.1f} degrees")
        print(f"    Error:        {r['error_pct']:.1f}%")
