"""
Three-Tier Parameter Census Table
===================================

Book reference: Chapter 38

Generates the full parameter census table showing
Tier 1 (structural), Tier 2 (tau postulate), Tier 3 (fitted).
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent / "parameter-census"))
from summary_table import generate_summary


def plot_parameter_census(output_path=None, figsize=(14, 8)):
    """Generate parameter census table as figure."""

    summary = generate_summary()

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    columns = ["Parameter", "Predicted", "Observed", "Accuracy", "Tier"]
    rows = []

    for entry in summary:
        rows.append([
            entry["name"],
            f"{entry['predicted']:.6g}",
            f"{entry['observed']:.6g}",
            entry["accuracy"],
            f"Tier {entry['tier']}",
        ])

    # Color by tier
    colors = []
    for entry in summary:
        if entry["tier"] == 1:
            colors.append(['#E3F2FD'] * len(columns))
        elif entry["tier"] == 2:
            colors.append(['#FFF3E0'] * len(columns))
        else:
            colors.append(['#F3E5F5'] * len(columns))

    table = ax.table(
        cellText=rows, colLabels=columns, loc='center',
        cellLoc='center', colColours=['#E0E0E0'] * len(columns),
        cellColours=colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    ax.set_title("Parameter Census: Three-Tier Derivation (Ch. 38)",
                 fontsize=14, fontweight='bold', pad=20)

    # Legend
    legend_text = ("Tier 1 (blue): Structural — no free inputs\n"
                   "Tier 2 (orange): Requires tau = 1/(2+phi)\n"
                   "Tier 3 (purple): One fitted parameter")
    ax.text(0.5, -0.02, legend_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=9, style='italic')

    plt.tight_layout()

    if output_path:
        for ext in ['.png', '.pdf']:
            fig.savefig(str(output_path).replace('.png', ext),
                        dpi=150, bbox_inches='tight')

    plt.close(fig)
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="parameter_census_table.png")
    args = parser.parse_args()
    plot_parameter_census(args.output)
    print(f"Generated {args.output}")
