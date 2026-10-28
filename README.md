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

Data files live under `$DATA_DIR` (defined in `config.py`), currently:
```
/global/cfs/cdirs/desi/users/smcole/DirectSHAM/6thOct/
```
with model-specific files in a subdirectory named by the model parameters,
e.g. `N_neigh_lum50/seed000_sig_lgv0.18_0.30_fc0.00_al0.00/`.

## Pipeline scripts

### Mock construction

| Script | Node | Description |
|--------|------|-------------|
| `make_random.py` | login | Constructs the fordata random catalogue using the Cole (2011) clone method with a ±Δz=0.03 redshift window and Metropolis–Hastings boundary correction |
| `make_checkpoint.py` | compute | Reads Uchuu subhaloes in parallel, computes cosmological and redshift-space redshifts, applies Vpeak-dependent scatter, masks to survey footprint, and writes a checkpoint file |
| `make_mock.py` | login | Runs shell-by-shell SHAM to match subhaloes to random catalogue galaxies, optionally with two-parameter colour matching via assembly history |
| `split_ngcsgc.py` | login | Splits mock, data and random catalogues into NGC/SGC by galactic latitude; updates FSKY metadata correctly for each cap |
| `inspect_mocks.py` | login | Diagnostic plots comparing mock and data distributions (sky, redshift, photometric, halo properties, jackknife dN/dz) |

### Stellar mass and SFR augmentation

| Script | Node | Description |
|--------|------|-------------|
| `add_stellar_masses.py` | login | Matches CIGALE and FSF VACs to a mock or BGS data catalogue via TARGETID; fills unmatched objects with donor-based stellar masses; writes `*_augmented.fits` output. Pass `--cat` to augment any catalogue (e.g. master BGS data file) instead of a mock |
| `vacmatching.py` | — | Library of donor-matching functions used by `add_stellar_masses.py`: `find_bad_rows`, `assign_donors_first_pass` (KD-tree in Zobs/Mr/colour space), `assign_donors_second_pass`, `add_provenance_flags` |
| `propagate_stellar_masses.py` | login | Propagates LOGM, LOGSFR, cigflag, LOGMSTAR, SFR, fsfflag from the master augmented BGS catalogue to a list of other files (mocks, randoms, NGC/SGC subsets) via TARGETID join. Writes in-place atomically |
| `inspect_stellar_masses.py` | login | Diagnostic plots for the stellar-mass augmented catalogues: CIGALE vs FSF M\*, Mr vs M\*, colour vs sSFR, provenance redshift distributions, M\* distributions in redshift shells |
| `plot_mstar_ssfr.py` | login | M\* vs sSFR hexbin plots for CIGALE and FSF, with a candidate SF/Q cut overlaid; used to define sSFR-based sample selections |
| `plot_nz_mstar.py` | login | Comoving number density n(z) [h³ Mpc⁻³] for Mr and M\* threshold samples; used to choose M\* cuts matched to Mr samples |
| `plot_cigale_chi2.py` | login | CHI2 distribution diagnostic: shows that CIGALE incompleteness is entirely from CHI2>2 (poor WISE photometry for faint dwarfs) |

### Clustering measurement

| Script | Node | Description |
|--------|------|-------------|
| `measure_clustering.py` | compute | Applies magnitude/colour/redshift cuts to data and randoms, then calls `xirunpc.py` with PIP inverse-bitwise weights and jackknife covariance |
| `measure_mock_clustering.py` | compute | Same as above for mocks, using unit weights and redshift-space positions (`Zobs`) |
| `measure_mock_clustering_colour.py` | compute | Red/blue colour-split version of mock clustering measurement |
| `compare_clustering.py` | login | Single-panel plot of ξ₀ and ξ₂ with jackknife error bars, for one or more mocks vs data |
| `compare_clustering_5panel.py` | login | Five-panel version covering the five standard magnitude-limited BGS samples |

### Orchestration

| Script | Node | Description |
|--------|------|-------------|
| `run_pipeline.sh` | compute | Runs checkpoint, mock construction, splitting and clustering for a single model (sig_lgv_ref, sig_lgv_slope) across all magnitude samples |

### Configuration

| File | Description |
|------|-------------|
| `config.py` | Central configuration: data paths, default parameters, cosmology |
| `pipeline_utils.py` | Shared utilities: `setup_logging()`, `make_paths()`, `write_fits_meta()` |

## Scatter parametrisation

Vpeak scatter is luminosity-dependent:

```
σ_v = σ_{v,0} + σ_{v,1} × (log₁₀ Vpeak − log₁₀ V_piv)
```

Current best-fit values: σ_{v,0} = 0.18, σ_{v,1} = 0.30, log₁₀(V_piv) = 2.6.

Assembly-bias colour matching uses two additional parameters:
- `f_colour`: scatter in log(Mpeak_Scale) before FracZpeak computation
- `alpha`: correlation between Vpeak scatter and FracZpeak rank

## Standard magnitude-limited samples

| Sample | Mr cut | zmin | zmax |
|--------|--------|------|------|
| 1 | −21.5 | 0.02 | 0.42 |
| 2 | −20.5 | 0.02 | 0.34 |
| 3 | −19.5 | 0.02 | 0.27 |
| 4 | −18.5 | 0.02 | 0.21 |
| 5 | −17.5 | 0.02 | 0.17 |

Colour split: Red if ⁰·¹(g−r) ≥ 0.21 − 0.03 × Mr, Blue otherwise.

## Stellar mass samples

Stellar masses from two VACs matched via TARGETID:
- **CIGALE**: `BGS_BRIGHT-loa-LSSv2.1-cigale-v1.0.fits` (v2.0, M. Siudek).
  Quality cuts: LOGM > 6, CHI2 < 2. ~3% of galaxies require donor matching.
- **FSF** (FastSpecFit): `BGS_ANY-y3gqp-parent-fsf-v1.0.fits`.
  FSF values corrected to h=1 units: LOGMSTAR += 2×log₁₀(1/0.674) = +0.343 dex.
  ~0.8% of galaxies require donor matching.

Use CIGALE for SF/Q separation (cigflag); FSF stellar masses are systematically
~0.36 dex higher than CIGALE due to different model assumptions.

Adopted stellar mass thresholds (matched to Mr samples above):

| Mr cut | CIGALE log M\* | FSF log M\* |
|--------|---------------|------------|
| −21.5  | > 11.10       | > 11.45    |
| −20.5  | > 10.60       | > 10.95    |
| −19.5  | > 10.10       | > 10.45    |
| −18.5  | >  9.50       | >  9.85    |
| −17.5  | >  8.90       | >  9.25    |

Provenance flags (cigflag / fsfflag): 1 = direct TARGETID match,
2 = first-pass KD-tree donor, 3 = second-pass nearest-redshift donor,
−1 = unresolved (NaN values).

## Computing environment

All work runs on NERSC Perlmutter. Login nodes are sufficient for most steps;
compute nodes are required for `make_checkpoint.py` and the clustering scripts.

```bash
# Allocate an interactive compute node
salloc --nodes=1 --qos=interactive --constraint=cpu --account=desi --mem=128GB --time=02:00:00

# Set up the DESI software environment
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONPATH=:$SCRATCH/LSS/py:$HOME/ShaunColeDESI:$PYTHONPATH
```

Useful bash functions (add to `~/.bashrc`):
```bash
# List column names in a FITS file
columns() { python -c "import fitsio; h=fitsio.read_header('$1',ext=1); [print(c['ttype']) for c in h.records() if 'ttype' in c]"; }

# Show metadata key-value pairs in a FITS file
fitsmeta() { python -c "
from astropy.io import fits
with fits.open('$1') as h:
    for ext in [0,1]:
        hdr = h[ext].header
        if len(hdr) <= 4: continue
        print(f'--- HDU {ext} ---')
        for k,v in hdr.items():
            if k not in ('','COMMENT','HISTORY'): print(f'  {k:<20s} {v}')
"; }
```

## Cosmology

All distances in Mpc/h using:
```python
FlatLambdaCDM(H0=100, Om0=0.313, Tcmb0=2.725)
```

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sig_lgv_ref` | 0.18 | log₁₀(Vpeak) scatter amplitude σ_{v,0} at pivot |
| `sig_lgv_slope` | 0.30 | Gradient σ_{v,1} of scatter with log₁₀(Vpeak) |
| `lgv_pivot` | 2.6 | log₁₀(V_piv) pivot point |
| `f_colour` | 0.0 | Assembly-bias colour scatter parameter σ_col |
| `alpha` | 0.0 | Correlated scatter parameter α_v |
| `obs_seed` | 0 | Observer position seed |
| `min_Vpeak` | 70 km/s | Minimum subhalo Vpeak |
| `N_neigh_vpeak` | 1000 | Neighbours for FracZpeak estimate |
| `N_neigh_lum` | 50 | Neighbours for colour matching |
| `nzbins` | 100 | Number of redshift shells for SHAM |
| `Δz` | 0.03 | Redshift window for random catalogue cloning |

## Typical workflow

```bash
cd ~/DirectSHAM
REF=0.18; SLOPE=0.30; FC=0.00; AL=0.00; SEED=0

# 1. Build the random catalogue (login node, once per data release)
python pipeline/make_random.py

# 2. Build the subhalo checkpoint (compute node, ~1 hr)
python pipeline/make_checkpoint.py --reg N --obs_seed $SEED \
    --sig_lgv_ref $REF --sig_lgv_slope $SLOPE
python pipeline/make_checkpoint.py --reg S --obs_seed $SEED \
    --sig_lgv_ref $REF --sig_lgv_slope $SLOPE

# 3. Run SHAM to produce mock (login node, N before S)
python pipeline/make_mock.py --reg N --obs_seed $SEED \
    --sig_lgv_ref $REF --sig_lgv_slope $SLOPE
python pipeline/make_mock.py --reg S --obs_seed $SEED \
    --sig_lgv_ref $REF --sig_lgv_slope $SLOPE

# 4. Split into NGC/SGC (login node)
python pipeline/split_ngcsgc.py --sig_lgv_ref $REF --sig_lgv_slope $SLOPE

# 5. Measure luminosity clustering for all 5 samples (compute node)
bash pipeline/run_all_clustering.sh

# 6. Compare with data
python pipeline/compare_clustering_5panel.py \
    --sig_lgv_ref $REF --sig_lgv_slope $SLOPE

# 7. Add stellar masses to master BGS catalogue (login node, once)
python pipeline/add_stellar_masses.py \
    --cat data/6thOct/BGS_BRIGHT_6thOct.fits \
    --cat_out data/6thOct/BGS_BRIGHT_6thOct_augmented.fits

# 8. Propagate stellar masses to mocks, randoms and data subsets
MASTER=data/6thOct/BGS_BRIGHT_6thOct_augmented.fits
MODEL=data/6thOct/N_neigh_lum50/seed000_sig_lgv${REF}_${SLOPE}_fc${FC}_al${AL}
python pipeline/propagate_stellar_masses.py \
    --master $MASTER \
    --files \
        data/6thOct/BGS_BRIGHT_6thOct_NGC.fits \
        data/6thOct/BGS_BRIGHT_6thOct_SGC.fits \
        data/6thOct/rancat_fordata_NGC.fits \
        data/6thOct/rancat_fordata_SGC.fits \
        data/6thOct/rancat_formock_NGC.fits \
        data/6thOct/rancat_formock_SGC.fits \
        $MODEL/mock_NGC.fits \
        $MODEL/mock_SGC.fits

# 9. Inspect stellar mass augmentation
python pipeline/inspect_stellar_masses.py --reg NGC SGC \
    --sig_lgv_ref $REF --sig_lgv_slope $SLOPE

# 10. Plot n(z) by stellar mass threshold
python pipeline/plot_nz_mstar.py \
    --sig_lgv_ref $REF --sig_lgv_slope $SLOPE \
    --mstar_cuts 11.10 10.60 10.10 9.50 8.90 \
    --mstar_cuts_fsf 11.45 10.95 10.45 9.85 9.25
```

## Data locations (NERSC)

| Item | Path |
|------|------|
| BGS DR2 data | `$DATA_DIR/6thOct/BGS_BRIGHT_6thOct.fits` |
| Fordata randoms | `$DATA_DIR/6thOct/rancat_fordata_{N,S,NGC,SGC}.fits` |
| Formock randoms | `$DATA_DIR/6thOct/rancat_formock_{N,S,NGC,SGC}.fits` |
| Mock catalogues | `$DATA_DIR/6thOct/N_neigh_lum50/seed000_sig_lgv0.18_0.30_fc0.00_al0.00/mock_{N,S,NGC,SGC}.fits` |
| Master augmented BGS | `$DATA_DIR/6thOct/BGS_BRIGHT_6thOct_augmented.fits` |
| Clustering output | `$DATA_DIR/6thOct/.../clustering/` |
| CIGALE VAC (v2.0) | `/global/cfs/cdirs/desi/science/gqp/y3gqp/v2.0/st_mass_MS_cigale/BGS_BRIGHT-loa-LSSv2.1-cigale-v1.0.fits` |
| FSF VAC | `/global/cfs/cdirs/desicollab/science/gqp/y3gqp/v1.0/parent/BGS_ANY-y3gqp-parent-fsf-v1.0.fits` |
| Uchuu halo catalogues | `/global/cfs/cdirs/desi/mocks/cai/Uchuu-SHAM/Uchuu-halo-catalogs/` |
| Log files | `~/DirectSHAM/logs/` |

## Dependencies

- DESI cosmodesi environment (`cosmodesi_environment.sh main`)
- astropy ≥ 7.2, numpy, scipy, matplotlib, fitsio
- pycorr 0.1.5
- LSS pipeline (`$SCRATCH/LSS/`)

## Authors

Shaun Cole (Durham ICC), with contributions from Elena Fernández-García (IAA Granada)

## References

- Cole S., 2011, MNRAS, 416, 739
- Ishiyama T. et al., 2021, MNRAS, 506, 4210
- Dong-Páez C. A. et al., 2024, MNRAS, 528, 7236
- Prada et al., 2025, A&A, 698, A170
- Fernández-García E. et al., 2026, JCAP, 2026, 002
- Moustakas J. et al., 2023 (FastSpecFit)
- Siudek M. et al., 2024 (CIGALE for DESI BGS)
