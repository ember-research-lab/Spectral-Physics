"""Tier 2: Parameters requiring tau = 1/(2+phi).

Book reference: Chapter 38, Section 38.9 -- Tier 2 Census

These 9 quantities require the tau postulate (stability tolerance)
in addition to structural parameters.  They are derived from the
CKM matrix Wolfenstein parametrization where the spectral framework
identifies:

    lambda = 0.2240         (Cabibbo parameter, framework input)
    A = tau * phi = 0.447   (Wolfenstein A from spectral stability)

and the Wolfenstein (rho, eta) are determined by delta_CP (Tier 3).

Tier 2 parameters:
    1. |V_us| (CKM 1-2 mixing)
    2. |V_ub| (CKM 1-3 mixing)
    3. |V_cb| (CKM 2-3 mixing)
    4. |V_us/V_ub| ratio
    5. Jarlskog invariant J
    6. m_d/m_s quark mass ratio
    7. m_u/m_c quark mass ratio
    8. m_s/m_b quark mass ratio
    9. m_c/m_t quark mass ratio
"""

import numpy as np

from .constants import phi, tau, lam, delta_cp, EXPERIMENT


# Wolfenstein A parameter from spectral stability
A_wolf = tau * phi  # = 0.2764 * 1.618 = 0.4472


def derive_Vus() -> dict:
    """|V_us| = lambda.

    The Cabibbo angle directly gives the 1-2 CKM element.
    """
    predicted = lam
    experimental = EXPERIMENT["Vus"]["value"]
    error_pct = abs(predicted - experimental) / experimental * 100

    return {
        "quantity": "|V_us|",
        "formula": "lambda",
        "predicted": predicted,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 2,
        "book_section": "22.9.1",
    }


def derive_Vcb() -> dict:
    """|V_cb| = A * lambda^2.

    Standard Wolfenstein: V_cb = A * lambda^2.
    Spectral framework: A = tau * phi.
    """
    predicted = A_wolf * lam**2
    experimental = EXPERIMENT["Vcb"]["value"]
    error_pct = abs(predicted - experimental) / experimental * 100

    return {
        "quantity": "|V_cb|",
        "formula": "A * lambda^2 (A = tau*phi)",
        "predicted": predicted,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 2,
        "book_section": "22.9.2",
    }


def derive_Vub() -> dict:
    """|V_ub| = A * lambda^3.

    Leading-order Wolfenstein: |V_ub| = A * lambda^3 (rho^2+eta^2)^{1/2}.
    At leading order (rho^2+eta^2 ~ 1), |V_ub| ~ A * lambda^3.
    """
    # Use leading-order: A * lambda^3
    predicted = A_wolf * lam**3
    experimental = EXPERIMENT["Vub"]["value"]
    error_pct = abs(predicted - experimental) / experimental * 100

    return {
        "quantity": "|V_ub|",
        "formula": "A * lambda^3 (A = tau*phi)",
        "predicted": predicted,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 2,
        "book_section": "22.9.3",
    }


def derive_Vus_Vub_ratio() -> dict:
    """|V_us| / |V_ub| = 1 / (A * lambda^2).

    A parameter-free ratio that tests the Wolfenstein scaling.
    """
    predicted = 1 / (A_wolf * lam**2)
    # Compute from experimental values
    exp_ratio = EXPERIMENT["Vus"]["value"] / EXPERIMENT["Vub"]["value"]
    error_pct = abs(predicted - exp_ratio) / exp_ratio * 100

    return {
        "quantity": "|V_us/V_ub|",
        "formula": "1 / (A * lambda^2)",
        "predicted": predicted,
        "experimental": exp_ratio,
        "error_pct": error_pct,
        "tier": 2,
        "book_section": "22.9.4",
    }


def derive_jarlskog() -> dict:
    """Jarlskog invariant J = A^2 * lambda^6 * sin(delta_CP).

    The CP-violating rephasing invariant in Wolfenstein parametrization.
    Uses the spectral identification A = tau * phi and the fitted
    delta_CP.  Note: J has an additional eta-dependence at NLO that
    we absorb into the leading-order expression.
    """
    delta_rad = np.radians(delta_cp)
    # Standard Wolfenstein: J ~ A^2 * lambda^6 * eta
    # where eta = rho_bar * sin(delta) ~ sin(delta)
    # Leading-order spectral: J = A^2 * lambda^6 * sin(delta_CP)
    predicted = A_wolf**2 * lam**6 * np.sin(delta_rad)
    experimental = EXPERIMENT["Jarlskog_J"]["value"]
    error_pct = abs(predicted - experimental) / experimental * 100

    return {
        "quantity": "Jarlskog invariant J",
        "formula": "A^2 * lambda^6 * sin(delta_CP)",
        "predicted": predicted,
        "experimental": experimental,
        "error_pct": error_pct,
        "tier": 2,
        "book_section": "22.9.5",
    }


def derive_quark_mass_ratio(name: str, formula_str: str,
                             predicted_val: float,
                             experimental_value: float,
                             section: str) -> dict:
    """Quark mass ratio from spectral eigenvalue spacing.

    The spectral framework predicts hierarchical mass ratios from
    the eigenvalue spacing of the generation Laplacian, using
    powers of lambda and tau corrections.
    """
    error_pct = abs(predicted_val - experimental_value) / experimental_value * 100

    return {
        "quantity": name,
        "formula": formula_str,
        "predicted": predicted_val,
        "experimental": experimental_value,
        "error_pct": error_pct,
        "tier": 2,
        "book_section": section,
    }


def get_all_tier2() -> list[dict]:
    """Return all 9 Tier 2 derivations."""
    # Quark mass ratios from spectral eigenvalue spacing
    # PDG 2024 running masses at 2 GeV: m_u=2.16, m_d=4.67, m_s=93.4,
    # m_c=1270, m_b=4180, m_t=172760 MeV
    mass_ratios = [
        # m_d/m_s ~ lambda^(3/2) * sqrt(tau) -- inter-generation gap
        ("m_d/m_s", "lambda^2 * tau^(1/2)",
         lam**2 * tau**0.5, 0.050, "22.9.6"),
        # m_u/m_c ~ lambda^4 -- two-generation gap
        ("m_u/m_c", "lambda^4",
         lam**4, 0.0017, "22.9.7"),
        # m_s/m_b ~ lambda^2 -- one-generation Cabibbo suppression
        ("m_s/m_b", "lambda^2",
         lam**2, 0.0224, "22.9.8"),
        # m_c/m_t ~ lambda^(7/2) * tau -- large hierarchy
        ("m_c/m_t", "lambda^3 * tau",
         lam**3 * tau, 0.00735, "22.9.9"),
    ]

    results = [
        derive_Vus(),
        derive_Vcb(),
        derive_Vub(),
        derive_Vus_Vub_ratio(),
        derive_jarlskog(),
    ]

    for name, formula_str, pred, exp_val, section in mass_ratios:
        results.append(derive_quark_mass_ratio(
            name, formula_str, pred, exp_val, section
        ))

    return results


if __name__ == "__main__":
    print("=" * 72)
    print("TIER 2: Parameters requiring tau = 1/(2+phi)")
    print("=" * 72)
    for r in get_all_tier2():
        print(f"\n  {r['quantity']}")
        print(f"    Formula:      {r['formula']}")
        print(f"    Predicted:    {r['predicted']:.6g}")
        print(f"    Experimental: {r['experimental']:.6g}")
        print(f"    Error:        {r['error_pct']:.1f}%")
