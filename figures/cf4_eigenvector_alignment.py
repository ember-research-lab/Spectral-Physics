"""
CF4++ Eigenvector Alignment Z-Scores
======================================

Book reference: Chapter 37

Plots eigenvector alignment z-scores for filament detection
in the CosmicFlows-4 density field.

Uses synthetic data for CI; real results in cosmology/results/.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent / "cosmology"))


def synthetic_alignment_zscores(N=100, n_bins=20, seed=42):
    """
    Generate synthetic alignment z-scores.

    Simulates eigenvector-filament alignment with signal in
    the first few eigenvectors and noise in the rest.
    """
    rng = np.random.RandomState(seed)

    eigenvector_indices = np.arange(1, N + 1)
    z_scores = np.zeros(N)

    # First ~5 eigenvectors show significant alignment
    z_scores[:5] = rng.uniform(2.5, 5.0, 5)
    # Next ~10 show marginal alignment
    z_scores[5:15] = rng.uniform(0.5, 2.0, 10)
    # Rest are noise
    z_scores[15:] = rng.randn(N - 15) * 0.5

    return eigenvector_indices, z_scores


def plot_eigenvector_alignment(output_path=None, figsize=(10, 5)):
    """Generate eigenvector alignment z-score plot."""

    indices, z_scores = synthetic_alignment_zscores()

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#4CAF50' if z > 2 else '#FF9800' if z > 1 else '#9E9E9E'
              for z in z_scores]

    ax.bar(indices[:30], z_scores[:30], color=colors[:30],
           edgecolor='black', linewidth=0.3)

    # Significance thresholds
    ax.axhline(2.0, color='red', linestyle='--', alpha=0.7, label='z = 2 (p < 0.05)')
    ax.axhline(3.0, color='darkred', linestyle=':', alpha=0.7, label='z = 3 (p < 0.003)')

    ax.set_xlabel('Eigenvector Index')
    ax.set_ylabel('Alignment z-score')
    ax.set_title('Laplacian Eigenvector-Filament Alignment (Ch. 37)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Annotation
    ax.text(0.95, 0.95,
            'Green: significant (z > 2)\nOrange: marginal\nGray: noise',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

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
    parser.add_argument("--output", default="cf4_eigenvector_alignment.png")
    args = parser.parse_args()
    plot_eigenvector_alignment(args.output)
    print(f"Generated {args.output}")
