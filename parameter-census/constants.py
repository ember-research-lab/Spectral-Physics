"""Framework parameters and experimental reference values.

Book reference: Chapter 38, Section 38.1 -- The Four Framework Parameters

The spectral framework has exactly four free parameters from which all
derived quantities follow.  Three (phi, tau, lambda) are structural;
delta_CP is the single fitted value.
"""

import numpy as np


# ── Framework parameters ─────────────────────────────────────────────────────

phi = (1 + np.sqrt(5)) / 2          # Golden ratio = 1.618033988749895
tau = 1 / (2 + phi)                  # Stability tolerance = 0.27639320225002106
lam = 0.2240                         # Cabibbo parameter (Wolfenstein lambda)
delta_cp = 68.75                     # CP-violating phase (degrees)


# ── Experimental reference values (PDG 2024 + Planck 2018) ──────────────────

EXPERIMENT = {
    # Tier 1: structural (no free inputs)
    "alpha": {
        "value": 1 / 137.035999177,   # fine-structure constant
        "uncertainty": 2.1e-10,
        "source": "PDG 2024",
    },
    "koide_K": {
        "value": 0.666611,             # Koide ratio from PDG lepton masses
        "uncertainty": 0.000003,
        "source": "PDG 2024 (me, mu, mtau)",
    },
    "sin2_theta_W": {
        "value": 0.23122,              # weak mixing angle (MS-bar, MZ)
        "uncertainty": 0.00003,
        "source": "PDG 2024",
    },
    "cosmological_ratio": {
        "value": 2.06,                 # Lambda / H0^2  (dimensionless)
        "uncertainty": 0.10,
        "source": "Planck 2018 + BAO",
    },
    "omega_DM": {
        "value": 0.265,                # dark-matter density parameter
        "uncertainty": 0.007,
        "source": "Planck 2018",
    },
    "omega_Lambda": {
        "value": 0.685,                # dark-energy density parameter
        "uncertainty": 0.007,
        "source": "Planck 2018",
    },
    "baryon_asymmetry": {
        "value": 6.1e-10,              # eta_B (baryon-to-photon ratio)
        "uncertainty": 0.04e-10,
        "source": "Planck 2018 + BBN",
    },
    "theta_13_PMNS": {
        "value": 8.61,                 # degrees
        "uncertainty": 0.13,
        "source": "PDG 2024 (Daya Bay + RENO + Double Chooz)",
    },
    "theta_12_PMNS": {
        "value": 33.41,                # degrees
        "uncertainty": 0.75,
        "source": "PDG 2024",
    },
    "theta_23_PMNS": {
        "value": 42.2,                 # degrees
        "uncertainty": 1.1,
        "source": "PDG 2024",
    },
    # CKM Wolfenstein parameters (for cross-check)
    "Vus": {
        "value": 0.2243,
        "uncertainty": 0.0005,
        "source": "PDG 2024",
    },
    "Vub": {
        "value": 0.00382,
        "uncertainty": 0.00020,
        "source": "PDG 2024",
    },
    "Vcb": {
        "value": 0.0408,
        "uncertainty": 0.0014,
        "source": "PDG 2024",
    },
    "Jarlskog_J": {
        "value": 3.08e-5,
        "uncertainty": 0.15e-5,
        "source": "PDG 2024",
    },
    # Proton lifetime (lower bound)
    "proton_lifetime_years": {
        "value": 1.6e34,               # lower bound, p -> e+ pi0
        "uncertainty": None,            # limit, not measurement
        "source": "Super-Kamiokande 2020",
    },
}


# ── Tier classification ──────────────────────────────────────────────────────

TIER_1_LABELS = [
    "alpha",
    "koide_K",
    "sin2_theta_W",
    "cosmological_ratio",
    "omega_DM",
    "omega_Lambda",
    "baryon_asymmetry",
    "theta_13_PMNS",
    "theta_12_PMNS",
    "theta_23_PMNS",
]

TIER_2_LABELS = [
    "Vus",
    "Vub",
    "Vcb",
    "Vus_Vub_ratio",
    "Jarlskog_J",
    "md_ms_ratio",
    "mu_mc_ratio",
    "ms_mb_ratio",
    "mc_mt_ratio",
]

TIER_3_LABELS = [
    "delta_cp",
]
