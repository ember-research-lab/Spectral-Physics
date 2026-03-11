# CF4++ Data

## CosmicFlows-4++ Mean/Std Grids

The cosmology analyses use the CosmicFlows-4++ density and velocity grids
(Courtois et al. 2023). This file is 161 MB and is NOT included in git.

### Download

1. Visit the Extragalactic Distance Database: https://edd.ifa.hawaii.edu/
2. Navigate to CF4++ data products
3. Download `CF4pp_mean_std_grids.npz`
4. Place it in this directory: `cosmology/data/CF4pp_mean_std_grids.npz`

### File Contents

The `.npz` file contains 128x128x128 grids covering a 1000 Mpc box:

| Field | Shape | Description |
|-------|-------|-------------|
| `d_mean_CF4pp` | (128, 128, 128) | Density contrast delta = (rho - rho_bar) / rho_bar |
| `d_std_CF4pp` | (128, 128, 128) | Density uncertainty |
| `v_mean_CF4pp` | (3, 128, 128, 128) | 3D peculiar velocity (km/s) |
| `v_std_CF4pp` | (3, 128, 128, 128) | Velocity uncertainty |
| `vr_mean_CF4pp` | (128, 128, 128) | Radial peculiar velocity (km/s) |
| `vr_std_CF4pp` | (128, 128, 128) | Radial velocity uncertainty |

### Grid Parameters

- Grid size: 128^3 cells
- Box size: 1000 Mpc (centered on Milky Way)
- Cell size: 7.8125 Mpc
- Coordinates: Supergalactic (SGX, SGY, SGZ)

### Without CF4++ Data

All tests use synthetic data and do not require this file.
The analysis scripts will print a helpful message if the data is missing.
Use `create_synthetic_universe()` from `data_loader.py` for development.

### Reference

Courtois, H. M., et al. (2023). "Cosmicflows-4." ApJ, 902, 145.
