# Parameter Census

Companion code for **Chapter 38: The Parameter Census** of *Spectral Physics: A Unified Framework*.

## Book Reference

| Section | Module | Description |
|---------|--------|-------------|
| 38.1 | `constants.py` | Four framework parameters (phi, tau, lambda, delta\_CP) |
| 38.2 | `fine_structure.py` | Fine-structure constant alpha = tau^2 / (4 phi^2) |
| 38.3 | `koide_formula.py` | Koide ratio K = 2/3 from circulant mass matrix |
| 38.4 | `weinberg_angle.py` | Weak mixing angle sin^2(theta\_W) = lambda |
| 38.5 | `cosmological_constant.py` | Cosmological ratio Lambda/H\_0^2 = 2 (spectral floor) |
| 38.6 | `baryon_asymmetry.py` | Baryon asymmetry eta\_B = lambda^14 |
| 38.7 | `coincidence_problem.py` | Omega\_DM = tau, Omega\_Lambda = 1 - tau |
| 38.8 | `tier1_structural.py` | All 10 Tier 1 parameters including PMNS angles |
| 38.9 | `tier2_tau_postulate.py` | CKM elements, Jarlskog J, quark mass ratios |
| 38.10 | `tier3_fitted.py` | Single fitted parameter: delta\_CP |
| 38.11 | `summary_table.py` | Full census table and JSON export |

## Three-Tier Classification

The spectral framework derives physical constants from four parameters. These are organized into three **falsifiability tiers** based on how many framework inputs each derivation requires:

### Tier 1: Structural (10 parameters, no free inputs)

These follow purely from the golden ratio phi, the stability tolerance tau = 1/(2+phi), and the Cabibbo parameter lambda = 0.2240. No fitting is performed. Every prediction is falsifiable.

| Quantity | Formula | Predicted | Experiment | Error |
|----------|---------|-----------|------------|-------|
| alpha (fine structure) | tau^2 / (4 phi^2) | 1/137.08 | 1/137.04 | 0.03% |
| Koide K | 2/3 (circulant) | 0.66667 | 0.66661 | 0.01% |
| sin^2(theta\_W) | lambda | 0.224 | 0.231 | 3.1% |
| Lambda/H\_0^2 | 2 (spectral floor) | 2.0 | 2.06 | 2.9% |
| Omega\_DM | tau | 0.276 | 0.265 | 4.3% |
| Omega\_Lambda | 1 - tau | 0.724 | 0.685 | 5.6% |
| eta\_B (baryon asymmetry) | lambda^14 | 8.0e-10 | 6.1e-10 | ~31% |
| PMNS theta\_13 | arcsin(lambda/sqrt(2)) | 9.1 deg | 8.6 deg | 5.8% |
| PMNS theta\_12 | arctan(1/phi) | 31.7 deg | 33.4 deg | 5.1% |
| PMNS theta\_23 | pi/4 - lambda^2/2 | 43.6 deg | 42.2 deg | 3.2% |

### Tier 2: Requires tau postulate (9 parameters)

These additionally use the spectral identification A = tau * phi for the Wolfenstein A parameter. They predict CKM matrix elements, the Jarlskog invariant, and quark mass hierarchies.

### Tier 3: Fitted (1 parameter)

The single fitted parameter is the CKM CP-violating phase delta\_CP = 68.75 degrees. The framework predicts that no second fitted parameter is needed.

## Falsifiable Prediction

The spectral framework makes one absolute structural prediction that is not yet tested:

**The proton is absolutely stable.**

Baryon number conservation is exact in the spectral framework because the Laplacian's block structure forbids transitions between the baryon and lepton sectors. There is no grand unified gauge boson to mediate proton decay. Current experimental bounds (tau\_p > 1.6 x 10^34 years, Super-Kamiokande) are consistent. Hyper-Kamiokande will extend this by an order of magnitude.

If proton decay is observed, the spectral framework is falsified.

## Usage

```bash
# Print the full census table and generate results/census.json
uv run python -m parameter-census.summary_table

# Run individual derivations
uv run python -m parameter-census.fine_structure
uv run python -m parameter-census.koide_formula
uv run python -m parameter-census.coincidence_problem

# Run tests
uv run pytest parameter-census/tests/ -v
```

## File Structure

```
parameter-census/
  __init__.py
  constants.py              # Framework parameters + experimental references
  fine_structure.py         # alpha = tau^2 / (4 phi^2)
  koide_formula.py          # K = 2/3 from circulant structure
  weinberg_angle.py         # sin^2(theta_W) = lambda
  cosmological_constant.py  # Lambda/H0^2 = 2
  baryon_asymmetry.py       # eta_B = lambda^14
  coincidence_problem.py    # Omega_DM = tau, Omega_Lambda = 1 - tau
  tier1_structural.py       # All 10 Tier 1 derivations
  tier2_tau_postulate.py    # All 9 Tier 2 derivations
  tier3_fitted.py           # Single fitted parameter
  summary_table.py          # Full table + JSON export
  results/
    census.json             # Reference results for regression tests
  tests/
    conftest.py             # Import path configuration
    test_census.py          # 58 tests covering all derivations
```
