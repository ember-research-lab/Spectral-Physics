# Figure Generation Scripts

Scripts that generate data-driven figures from the book.
Each script generates both PNG and PDF output.

## Usage

```bash
# Generate all figures
for f in *.py; do uv run python "$f"; done

# Generate specific figure
uv run python parameter_census_table.py --output output/
```

## Files

| Script | Book Reference | Description |
|--------|---------------|-------------|
| `eigenstate_decomposition.py` | Ch. 37 | Universe eigenstate bar chart |
| `parameter_census_table.py` | Ch. 38 | Three-tier parameter census |
| `cf4_eigenvector_alignment.py` | Ch. 37 | Filament alignment z-scores |
| `spectral_early_universe.py` | Ch. 37 | Mode amplitudes + retrodiction |
| `robustness_configurations.py` | Ch. 37 | Config sweep stability |
