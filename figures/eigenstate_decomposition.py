"""
Universe Eigenstate Decomposition Bar Chart
=============================================

Book reference: Chapter 37

Shows the decomposition of the universe state into
Laplacian eigenstates: c_0, c_1, c_2.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent / "cosmology"))
from eigenstate import compute_eigenstate_coefficients


def plot_eigenstate_decomposition(output_path=None, figsize=(8, 5)):
    """Generate eigenstate decomposition bar chart."""

    coeffs = compute_eigenstate_coefficients()

    fig, ax = plt.subplots(figsize=figsize)

    labels = [f"$c_{i}$" for i in range(len(coeffs["coefficients"]))]
    values = coeffs["coefficients"]
    colors = ['#2196F3', '#FF9800', '#4CAF50'][:len(values)]

    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('Coefficient Magnitude')
    ax.set_title('Universe Eigenstate Decomposition (Ch. 37)')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, axis='y', alpha=0.3)

    # Annotation
    ax.text(0.95, 0.95, f"$|\\Psi\\rangle = \\sum_k c_k |\\phi_k\\rangle$",
            transform=ax.transAxes, ha='right', va='top', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
    parser.add_argument("--output", default="eigenstate_decomposition.png")
    args = parser.parse_args()
    plot_eigenstate_decomposition(args.output)
    print(f"Generated {args.output}")
