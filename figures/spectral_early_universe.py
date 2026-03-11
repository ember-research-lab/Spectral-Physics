"""
Spectral Early Universe: Mode Amplitudes and Retrodiction
==========================================================

Book reference: Chapter 37

Shows heat kernel retrodiction of primordial density field
mode amplitudes from present-day observations.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


def synthetic_mode_amplitudes(N=50, seed=42):
    """
    Generate synthetic mode amplitudes for present and retrodicted past.

    Models the heat kernel: a_k(0) = a_k(t) * exp(lambda_k * t)
    where lambda_k are Laplacian eigenvalues.
    """
    rng = np.random.RandomState(seed)

    # Eigenvalues (roughly matching spectral gap structure)
    eigenvalues = np.sort(rng.exponential(scale=0.5, size=N))
    eigenvalues[0] = 0  # Zero mode

    # Present-day amplitudes (power-law decay)
    k = np.arange(1, N + 1)
    present_amplitudes = 1.0 / (k ** 0.8) + rng.randn(N) * 0.02

    # Retrodicted amplitudes (inverse heat kernel)
    t = 1.0  # Diffusion time
    retrodicted = present_amplitudes * np.exp(eigenvalues * t)

    return {
        "eigenvalues": eigenvalues,
        "present": present_amplitudes,
        "retrodicted": retrodicted,
        "modes": k,
    }


def plot_early_universe(output_path=None, figsize=(12, 5)):
    """Generate early universe retrodiction figure."""

    data = synthetic_mode_amplitudes()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Mode amplitudes comparison
    ax1.semilogy(data["modes"][:30], np.abs(data["present"][:30]),
                 'b-o', markersize=4, label='Present (observed)')
    ax1.semilogy(data["modes"][:30], np.abs(data["retrodicted"][:30]),
                 'r-s', markersize=4, label='Retrodicted (t=0)')
    ax1.set_xlabel('Mode Index k')
    ax1.set_ylabel('|a_k| (log scale)')
    ax1.set_title('Mode Amplitudes: Present vs Retrodicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Amplification factor
    ratio = data["retrodicted"] / np.maximum(np.abs(data["present"]), 1e-10)
    ax2.semilogy(data["modes"][:30], ratio[:30], 'g-^', markersize=4)
    ax2.set_xlabel('Mode Index k')
    ax2.set_ylabel('Amplification Factor')
    ax2.set_title(r'$e^{\lambda_k t}$ Amplification')
    ax2.grid(True, alpha=0.3)

    # Annotation about Tikhonov regularization
    ax2.text(0.95, 0.95,
             'High-k modes amplified\n→ Tikhonov regularization\nrequired for stability',
             transform=ax2.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Heat Kernel Retrodiction of Primordial Density (Ch. 37)',
                 fontsize=13, fontweight='bold')
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
    parser.add_argument("--output", default="spectral_early_universe.png")
    args = parser.parse_args()
    plot_early_universe(args.output)
    print(f"Generated {args.output}")
