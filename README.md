# DirectSHAM

Pipeline for constructing mock DESI BGS catalogues from the Uchuu simulation
using direct subhalo abundance matching (SHAM), and measuring their
two-point clustering statistics.

## Overview

The pipeline populates Uchuu dark matter subhaloes with observed DESI BGS
galaxies using a modified SHAM procedure that directly inherits photometric
properties from the data. The resulting mock catalogues reproduce the BGS
selection function by construction and can be compared directly with the data
using the same analysis tools.

The method is described in Cole et al. (2026, in preparation).

## Structure

```
DirectSHAM/
├── pipeline/                   # Core Python scripts (see below)
├── notebooks/                  # Exploratory and visualisation notebooks
├── logs/                       # Log files (not tracked by git)
└── README.md
```

## Pipeline scripts

### Mock construction

| Script | Node | Description |
|--------|------|-------------|
| `make_random.py` | login | Constructs the fordata random catalogue using the Cole (2011) clone method with a ±Δz=0.03 redshift window and Metropolis–Hastings boundary correction |
| `make_checkpoint.py` | compute | Reads Uchuu subhaloes in parallel, computes cosmological and redshift-space redshifts, applies Vpeak scatter, masks to survey footprint, and writes a checkpoint file |
| `make_mock.py` | login | Runs shell-by-shell SHAM to match subhaloes to random catalogue galaxies, optionally with two-parameter colour matching via assembly history |
| `split_ngcsgc.py` | login | Splits mock, data and random catalogues into NGC/SGC by galactic latitude and North/South photometric regions |
| `inspect_mocks.py` | login | Diagnostic plots comparing mock and data distributions (sky, redshift, photometric, halo properties, jackknife dN/dz) |

### Clustering measurement

| Script | Node | Description |
|--------|------|-------------|
| `measure_clustering.py` | compute | Applies magnitude and redshift cuts to data and randoms, then calls `xirunpc.py` with PIP inverse-bitwise weights and optional jackknife covariance |
| `measure_mock_clustering.py` | compute | Same as above for mocks, using unit weights and redshift-space positions (`Zobs`) |
| `compare_clustering.py` | login | Single-panel plot of ξ₀ and ξ₂ with jackknife error bars, for one or more mocks vs data |
| `compare_clustering_5panel.py` | login | Five-panel version covering the five standard magnitude-limited BGS samples |

### Orchestration

| Script | Node | Description |
|--------|------|-------------|
| `run_all_clustering.sh` | compute | Runs data and mock clustering sequentially for all 5 magnitude-limited samples |
| `run_sig_lgv_scan.sh` | compute | Runs checkpoint, mock construction and clustering for a list of `sig_lgv` scatter values across all 5 samples |

### Configuration

| File | Description |
|------|-------------|
| `config.py` | Central configuration: data paths, default parameters |
| `pipeline_utils.py` | Shared utilities: logging setup, path construction |

## Standard magnitude-limited samples

| Sample | Mr cut | zmax |
|--------|--------|------|
| 1 | −22.0 | 0.50 |
| 2 | −21.5 | 0.42 |
| 3 | −21.0 | 0.34 |
| 4 | −20.5 | 0.28 |
| 5 | −20.0 | 0.23 |

All samples use zmin = 0.02.

## Computing environment

All work runs on NERSC Perlmutter. Login nodes are sufficient for
most steps; compute nodes are required for `make_checkpoint.py` and
the clustering measurement scripts.

```bash
# Allocate an interactive compute node (4 hours)
salloc --nodes=1 --qos=interactive --constraint=cpu --account=desi --time=04:00:00

# Set up the DESI software environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$HOME/ShaunColeDESI:$PYTHONPATH
```

## Cosmology

All distances in Mpc/h using:
```python
FlatLambdaCDM(H0=100, Om0=0.313, Tcmb0=2.725)
```

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sig_lgv` | 0.04 | log₁₀(Vpeak) scatter in dex |
| `obs_seed` | 0 | Observer position seed |
| `min_Vpeak` | 70 km/s | Minimum subhalo Vpeak |
| `N_neigh_vpeak` | 1000 | Neighbours for FracZpeak estimate |
| `N_neigh_lum` | 50 | Neighbours for colour matching |
| `nzbins` | 100 | Number of redshift shells for SHAM |
| `Δz` | 0.03 | Redshift window for random catalogue cloning |
| `njack` | 60 | Jackknife regions for data clustering errors |

## Typical workflow

```bash
# 1. Build the random catalogue (login node)
python make_random.py

# 2. Build the subhalo checkpoint (compute node)
python make_checkpoint.py --reg N --obs_seed 0 --sig_lgv 0.04
python make_checkpoint.py --reg S --obs_seed 0 --sig_lgv 0.04

# 3. Run SHAM to produce mock (login node)
python make_mock.py --obs_seed 0 --sig_lgv 0.04

# 4. Split into NGC/SGC (login node)
python split_ngcsgc.py --obs_seed 0 --sig_lgv 0.04

# 5. Measure clustering for all 5 samples (compute node)
bash run_all_clustering.sh

# 6. Compare with data
python compare_clustering_5panel.py --sig_lgv 0.04 --use_jackknife \
    --savefig 5panel_comparison.png

# 7. Scan scatter values (compute node)
bash run_sig_lgv_scan.sh
```

## Data locations (NERSC)

| Item | Path |
|------|------|
| BGS DR2 data | `~/ShaunColeDESI/sdata/Mocks/` |
| Uchuu halo catalogues | `/global/cfs/cdirs/desi/mocks/cai/Uchuu-SHAM/Uchuu-halo-catalogs/` |
| Clustering output | `$DATA_DIR/clustering/` |
| Log files | `~/DirectSHAM/logs/` |

## Dependencies

- DESI cosmodesi environment (`cosmodesi_environment.sh main`)
- pycorr 0.1.5
- astropy, numpy, scipy, matplotlib, fitsio
- LSS pipeline (`$SCRATCH/LSS/`)

## Authors

Shaun Cole (Durham ICC), with contributions from Elena Fernández-García (IAA Granada)

## References

- Cole S., 2011, MNRAS, 416, 739
- Ishiyama T. et al., 2021, MNRAS, 506, 4210
- Dong-Páez C. A. et al., 2024, MNRAS, 528, 7236
- Prada et al., 2025, A&A, 698, A170
- Fernández-García E. et al., 2026, JCAP, 2026, 002
