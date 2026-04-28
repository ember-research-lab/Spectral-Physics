# Spectral Physics

Companion code for *Spectral Physics: A Unified Framework* (Ember Research Lab).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18961252.svg)](https://zenodo.org/records/18961252)

**Epistemic status:** These scripts reproduce the computational results reported in the book. They do not constitute independent validation. Every table, figure, and numerical result in the book is traceable to a script in this repository.

## Directory Map

| Directory | Book Chapter | Description |
|-----------|-------------|-------------|
| `parameter-census/` | Ch. 38 (The Parameter Census) | Derived physical constants from spectral framework |
| `cosmology/` | Ch. 37 (Cosmology) | CosmicFlows-4 analysis, universe eigenstate, spectral floor |
| `figures/` | Various | Scripts generating data-driven book figures |

## Installation

```bash
# Preferred (uv)
uv sync

# Or with pip
pip install -e .
```

### Optional dependencies

```bash
uv sync --extra cosmology   # astropy for cosmology
```

## Quick Start

```bash
# Reproduce the parameter census (Ch. 38) -- no external data needed
uv run python parameter-census/summary_table.py

# Run all tests
uv run pytest -v
```

## Data

Some subdirectories require external data:

- **cosmology/**: CosmicFlows-4++ grids (161 MB). See `cosmology/data/README.md` for download instructions.

## Citation

```bibtex
@misc{benshalom2026spectral,
  title     = {Spectral Physics: A Unified Framework},
  author    = {Ben-Shalom, Aaron},
  year      = {2026},
  doi       = {10.5281/zenodo.18946321},
  publisher = {Zenodo},
  note      = {Preprint, v0.8}
}
```

## License

Code: MIT. Documentation: CC-BY-4.0.
