"""Generate the full three-tier parameter census table.

Book reference: Chapter 38, Section 38.11 -- Summary Census

Prints a formatted table of all derived quantities across all three
tiers and writes the results to results/census.json for regression
testing.
"""

import json
import os
import sys

from .tier1_structural import get_all_tier1
from .tier2_tau_postulate import get_all_tier2
from .tier3_fitted import get_all_tier3


def collect_all() -> list[dict]:
    """Collect all derivations across all tiers."""
    return get_all_tier1() + get_all_tier2() + get_all_tier3()


def format_value(v: float) -> str:
    """Format a numerical value for table display."""
    if abs(v) < 1e-4:
        return f"{v:.2e}"
    elif abs(v) < 0.01:
        return f"{v:.6f}"
    elif abs(v) < 10:
        return f"{v:.5f}"
    else:
        return f"{v:.2f}"


def print_table(results: list[dict]) -> None:
    """Print a formatted census table to stdout."""
    # Header
    print()
    print("=" * 90)
    print("PARAMETER CENSUS -- Chapter 38")
    print("Spectral Physics: A Unified Framework")
    print("=" * 90)

    current_tier = None
    tier_names = {
        1: "TIER 1: Structural (no free inputs)",
        2: "TIER 2: Requires tau = 1/(2+phi)",
        3: "TIER 3: Fitted",
    }

    header = (
        f"{'Quantity':<28} {'Formula':<30} "
        f"{'Theory':>12} {'Expt':>12} {'Error':>8}"
    )
    separator = "-" * 90

    for r in results:
        tier = r["tier"]
        if tier != current_tier:
            current_tier = tier
            print(f"\n  {tier_names[tier]}")
            print(f"  {separator}")
            print(f"  {header}")
            print(f"  {separator}")

        qty = r["quantity"][:27]
        formula = r["formula"][:29]
        theory = format_value(r["predicted"])
        expt = format_value(r["experimental"])

        # Format error with ~ for large errors
        if r["error_pct"] > 20:
            err = f"~{r['error_pct']:.0f}%"
        else:
            err = f"{r['error_pct']:.2f}%"

        print(f"  {qty:<28} {formula:<30} {theory:>12} {expt:>12} {err:>8}")

    print(f"\n  {separator}")
    print(f"  Total: {len(results)} parameters "
          f"({sum(1 for r in results if r['tier']==1)} structural, "
          f"{sum(1 for r in results if r['tier']==2)} tau-dependent, "
          f"{sum(1 for r in results if r['tier']==3)} fitted)")
    print()


def write_json(results: list[dict], path: str) -> None:
    """Write census results to JSON for regression testing."""
    # Round floats for stable comparison
    serializable = []
    for r in results:
        entry = dict(r)
        entry["predicted"] = float(r["predicted"])
        entry["experimental"] = float(r["experimental"])
        entry["error_pct"] = float(r["error_pct"])
        serializable.append(entry)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results written to {path}")


def main() -> None:
    """Run the full census."""
    results = collect_all()
    print_table(results)

    # Write JSON to results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "results", "census.json")
    write_json(results, json_path)


if __name__ == "__main__":
    main()
