"""Fine-structure constant from the spectral framework.

Book reference: Chapter 38, Section 38.2 -- The Fine-Structure Constant

Derivation
----------
The fine-structure constant emerges as the ratio of the stability
tolerance squared to four times the golden ratio squared:

    alpha = tau^2 / (4 * phi^2)

where tau = 1/(2+phi) is the spectral stability tolerance and
phi = (1+sqrt(5))/2 is the golden ratio.

Physical interpretation: alpha measures the strength of electromagnetic
coupling, which in the spectral framework corresponds to the projection
weight of the U(1) sector of the Laplacian onto the stability cone.
"""

import numpy as np

from .constants import phi, tau, EXPERIMENT


def derive_alpha() -> dict:
    """Derive the fine-structure constant.

    Returns
    -------
    dict with keys: predicted, experimental, error_pct, formula
    """
    predicted = tau**2 / (4 * phi**2)
    experimental = EXPERIMENT["alpha"]["value"]
    error_pct = abs(predicted - experimental) / experimental * 100

    return {
        "quantity": "Fine-structure constant alpha",
        "formula": "tau^2 / (4 * phi^2)",
        "predicted": predicted,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 1,
        "book_section": "22.2",
    }


if __name__ == "__main__":
    result = derive_alpha()
    print(f"Fine-structure constant alpha")
    print(f"  Formula:      {result['formula']}")
    print(f"  Predicted:    {result['predicted']:.7f}")
    print(f"  1/alpha:      {1/result['predicted']:.3f}")
    print(f"  Experimental: {result['experimental']:.7f}")
    print(f"  1/alpha_exp:  {1/result['experimental']:.3f}")
    print(f"  Error:        {result['error_pct']:.3f}%")
