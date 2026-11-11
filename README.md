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

The method is described in Cole et al. (2026, MNRAS submitted).

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

Lightcone model files live in a parallel directory:
```
/global/cfs/cdirs/desi/users/smcole/DirectSHAM/6thOct/lightcone_sig_lgv0.18_0.30_fc0.00_al0.00/
```

## Pipeline scripts

### Mock construction (cut-sky)

| Script | Node | Description |
|--------|------|-------------|
| `make_random.py` | login | Constructs the fordata random catalogue using the Cole (2011) clone method with a ±Δz=0.03 redshift window and Metropolis–Hastings boundary correction |
| `make_checkpoint.py` | compute | Reads Uchuu subhaloes in parallel, computes cosmological and redshift-space redshifts, applies Vpeak-dependent scatter, masks to survey footprint, and writes a checkpoint file |
| `make_mock.py` | login | Runs shell-by-shell SHAM to match subhaloes to random catalogue galaxies, optionally with two-parameter colour matching via assembly history |
| `split_ngcsgc.py` | login | Splits mock, data and random catalogues into NGC/SGC by galactic latitude; updates FSKY metadata correctly for each cap |
| `inspect_mocks.py` | login | Diagnostic plots comparing mock and data distributions (sky, redshift, photometric, halo properties, jackknife dN/dz) |

### Lightcone mock construction

Produces an onion-shell lightcone mock by stacking cut-sky mocks from six
Uchuu snapshots.  The observer is always at the box centre (obs_seed=0);
all snapshots share this position so the geometry is consistent.  Shell
boundaries are applied to the real-space redshift Z (not Zobs).

| Script | Node | Description |
|--------|------|-------------|
| `make_checkpoint_snap.py` | compute | As `make_checkpoint.py` but takes `--snapshot` to select the Uchuu snapshot; writes to the lightcone model directory. Run once per snapshot (~30–60 min each). |
| `make_mock_snap.py` | login | As `make_mock.py` but reads from the lightcone checkpoint and writes a full-redshift-range shell mock per snapshot. |
| `make_lightcone.py` | login | Reads all six shell mocks, applies real-space redshift cuts, vstacks them, splits into NGC/SGC, and writes the final lightcone catalogues. |
| `slurm_lightcone_checkpoint.sh` | SLURM | Job array (tasks 0–5) running `make_checkpoint_snap.py` for all six snapshots in parallel. |
| `slurm_lightcone_mock.sh` | SLURM | Job array (tasks 0–11) running `make_mock_snap.py` for all six snapshots × two regions. |

**Lightcone shells** (from `LIGHTCONE_SHELLS` in `config.py`):

| Shell | z range | Snapshot | z_snap |
|-------|---------|----------|--------|
| 0 | 0.00–0.05 | 50 | 0.00 |
| 1 | 0.05–0.10 | 47 | 0.09 |
| 2 | 0.10–0.25 | 45 | 0.19 |
| 3 | 0.25–0.40 | 43 | 0.30 |
| 4 | 0.40–0.45 | 41 | 0.43 |
| 5 | 0.45–0.50 | 40 | 0.49 |

**Workflow:**
```bash
REF=0.18; SLOPE=0.30; FC=0.00; AL=0.00

# 1. Run 6 checkpoint jobs (SLURM job array, ~1 hr each)
#    Snap45 can be symlinked from the existing cut-sky checkpoint instead.
sbatch pipeline/slurm_lightcone_checkpoint.sh

# 2. Run 12 mock jobs after checkpoints complete
sbatch --dependency=afterok:<jobid> pipeline/slurm_lightcone_mock.sh

# 3. Assemble final lightcone catalogues (login node)
python pipeline/make_lightcone.py \
    --sig_lgv_ref $REF --sig_lgv_slope $SLOPE --f_colour $FC --alpha $AL
```

**Snap45 shortcut:** since the cut-sky pipeline already produced a checkpoint
for snapshot 45, you can symlink it into the lightcone directory rather than
rerunning the expensive checkpoint job:
```bash
LC=data/6thOct/lightcone_sig_lgv0.18_0.30_fc0.00_al0.00
CS=data/6thOct/N_neigh_lum50/seed000_sig_lgv0.18_0.30_fc0.00_al0.00
mkdir -p $LC
ln -s $PWD/$CS/subhalos_N_70.0_N_neigh_vpeak1000.fits $LC/subhalos_N_snap45.fits
ln -s $PWD/$CS/subhalos_S_70.0_N_neigh_vpeak1000.fits $LC/subhalos_S_snap45.fits
```
Then cancel SLURM array task 2 before submitting the checkpoint job array.

### Stellar mass and SFR augmentation

| Script | Node | Description |
|--------|------|-------------|
| `add_stellar_masses.py` | login | Matches CIGALE and FSF VACs to a mock or BGS data catalogue via TARGETID; fills unmatched objects with donor-based stellar masses; writes `*_augmented.fits` output. Pass `--cat` to augment any catalogue (e.g. master BGS data file) instead of a mock |
| `vacmatching.py` | — | Library of donor-matching functions used by `add_stellar_masses.py`: `find_bad_rows`, `assign_donors_first_pass` (KD-tree in Zobs/Mr/colour space), `assign_donors_second_pass`, `add_provenance_flags` |
| `propagate_stellar_masses.py` | login | Propagates LOGM, LOGSFR, cigflag, LOGMSTAR, SFR, fsfflag from the master augmented BGS catalogue to a list of other files (mocks, randoms, NGC/SGC subsets) via TARGETID join. Writes in-place atomically |
| `inspect_stellar_masses.py` | login | Diagnostic plots for the stellar-mass augmented catalogues: CIGALE vs FSF M*, Mr vs M*, colour vs sSFR, provenance redshift distributions, M* distributions in redshift shells. **Paper: Fig. 12** |
| `plot_mstar_ssfr.py` | login | M* vs sSFR hexbin plots for CIGALE and FSF, with a candidate SF/Q cut overlaid; used to define sSFR-based sample selections |
| `plot_nz_mstar.py` | login | Comoving number density n(z) [h³ Mpc⁻³] for Mr and M* threshold samples; used to choose M* cuts matched to Mr samples. **Paper: Fig. 13** |
| `plot_cigale_chi2.py` | login | CHI2 distribution diagnostic: shows that CIGALE incompleteness is entirely from CHI2>2 (poor WISE photometry for faint dwarfs) |
| `confusion_matrix_sfq.py` | login | Computes and displays the confusion matrix between the SF/Q (CIGALE sSFR) and red/blue (colour) classifications in stellar mass bins |

### Clustering measurement

| Script | Node | Description |
|--------|------|-------------|
| `measure_clustering.py` | compute | Applies magnitude/colour/redshift cuts to data and randoms, then calls `xirunpc.py` with PIP inverse-bitwise weights (128 realisations) and jackknife covariance (njack=60) |
| `measure_mock_clustering.py` | compute | Same as above for mocks, using unit weights and redshift-space positions (`Zobs`); no jackknife (njack=0) |
| `measure_mock_clustering_colour.py` | compute | Red/blue colour-split version of mock clustering measurement |
| `measure_clustering_mstar.py` | compute | Data and mock xi for Mr/CIGALE/FSF stellar mass threshold samples; `--sfq {sf,passive}` flag applies log(sSFR)=-11 cut to galaxies AND randoms |
| `run_sfq_clustering.sh` | compute | Drives all 20 SF/passive clustering runs (5 samples × 2 splits × data+mock) |
| `measure_lss_clustering.py` | compute | Clustering for the LSS large-scale sample (Mr_LSS - 5log h < -20.7, 0.1 < z < 0.3; no K- or e-correction; DM from DESI tabulated cosmo). Writes output under `clustering/lss/`. Use `--cutsky` flag for cut-sky mock instead of lightcone. Custom bin edges via `.npz` file: s in [0.01, 150] Mpc/h, 100 log-spaced bins |

### Visualisation (paper figures)

| Script | Paper figure(s) | Description |
|--------|----------------|-------------|
| `compare_clustering_vpeak_scatter.py` | Fig. 8 | ξ₀ and ξ₂ vs data for five constant-σ_v models; five-panel layout by luminosity threshold |
| `S5PanelClustering2.py` | Fig. 9 | Same as above for the tuned luminosity-dependent scatter model (σ_{v,0}=0.18, σ_{v,1}=0.30) |
| `compare_clustering_colour.py` | Fig. 10 | Red/blue ξ₀ and ξ₂ for data and mock across five luminosity bins |
| `plot_linear_bias.py` | Fig. 11 | Linear bias b vs Mr threshold for full, red and blue samples (data circles, mock squares) |
| `compare_clustering_mstar.py` | Fig. 14 | Five-panel ξ₀ and ξ₂ for CIGALE and FSF stellar mass threshold samples; CIGALE red, FSF green |
| `plot_sfq_clustering.py` | Fig. 15 | ξ₀ for star-forming and quenched subsamples in four stellar mass bins (CIGALE sSFR cut) |
| `measure_lightcone_clustering.py` | Fig. 16 | Clustering measurement for the lightcone mock; output under `lightcone_dir/clustering/` |
| `plot_lss_clustering.py` | Fig. 17 | LSS sample ξ₀ and ξ₂: data (black/grey), lightcone mock (red/pink), cut-sky mock (blue, `--cutsky`); x-axis 0.5–150 Mpc/h |
| `compare_clustering.py` | — | Single-panel ξ₀ and ξ₂ for one or more mocks vs data; general-purpose diagnostic |
| `compare_clustering_5panel.py` | — | Five-panel diagnostic version of above |

### Orchestration

| Script | Node | Description |
|--------|------|-------------|
| `run_pipeline.sh` | compute | Runs checkpoint, mock construction, splitting and clustering for a single model (sig_lgv_ref, sig_lgv_slope) across all magnitude samples |
| `run_sfq_clustering.sh` | compute | Drives all 20 SF/passive clustering runs (5 samples × 2 splits × data+mock) |

### Configuration

| File | Description |
|------|-------------|
| `config.py` | Central configuration: data paths, default parameters, cosmology, lightcone shell definitions |
| `pipeline_utils.py` | Shared utilities: `setup_logging()`, `make_paths()`, `make_lightcone_paths()`, `write_fits_meta()` |
| `MockCode.py` | Core SHAM engine: shell-by-shell matching, FracZpeak computation, two-parameter colour matching |

## Scatter parametrisation

Vpeak scatter is luminosity-dependent:

```
σ_v = σ_{v,0} + σ_{v,1} × (log₁₀ Vpeak − log₁₀ V_piv)
```

Current best-fit values: σ_{v,0} = 0.18, σ_{v,1} = 0.30, log₁₀(V_piv) = 2.6.

Assembly-bias colour matching uses two additional parameters (both currently
set to zero in the published model; tuning is ongoing):

- `f_colour` (σ_col): scatter in log(Mpeak_Scale) before FracZpeak computation.
  Increasing σ_col dilutes the colour–assembly bias connection, reducing the
  clustering difference between red and blue galaxies.
- `alpha` (α_v): correlation between Vpeak scatter and FracZpeak rank.
  When α_v > 0, early-forming haloes receive a boost to their scattered Ṽ_peak,
  preferentially matching them to brighter galaxies and increasing the overall
  clustering amplitude of faint samples.

The scatter ε is drawn as:
```
ε = σ_v [ α_v · x_col,proxy + sqrt(1 - α_v²) · η ],   η ~ N(0,1)
x_col,proxy = sqrt(3)(2 f_col,proxy - 1)   (zero mean, unit variance)
```

## Standard magnitude-limited samples

| Sample | Mr cut | zmin | zmax |
|--------|--------|------|------|
| 1 | −21.5 | 0.02 | 0.42 |
| 2 | −20.5 | 0.02 | 0.28 |
| 3 | −19.5 | 0.02 | 0.19 |
| 4 | −18.5 | 0.02 | 0.12 |
| 5 | −17.5 | 0.02 | 0.08 |

Colour split: Red if ⁰·¹(g−r) ≥ 0.21 − 0.03 × Mr, Blue otherwise.

LSS large-scale sample: Mr_LSS - 5 log h < -20.7, 0.1 ≤ z ≤ 0.3.
No K-correction or e-correction; DM from DESI tabulated cosmo (equivalent to
Uchuu cosmology, Om0=0.3089). Used for BAO-scale clustering validation.

## Stellar mass samples

Stellar masses from two VACs matched via TARGETID:
- **CIGALE**: `BGS_BRIGHT-loa-LSSv2.1-cigale-v1.0.fits` (v2.0, M. Siudek).
  Quality cuts: LOGM > 6, CHI2 < 2. ~3% of galaxies require donor matching.
  Assumes Chabrier IMF, BC03 SPS models, WMAP7 cosmology (h=0.704).
- **FSF** (FastSpecFit): `BGS_ANY-y3gqp-parent-fsf-v1.0.fits`.
  FSF values corrected to h=1 units: LOGMSTAR += 2×log₁₀(1/0.6766) ≈ +0.343 dex.
  ~0.8% of galaxies require donor matching.
  Assumes Chabrier IMF, FSPS SPS models, Planck18 cosmology (h=0.6766).

CIGALE–FSF systematic offset: ~+0.35 dex (CIGALE higher), driven by different
SPS models and dust treatment. The h-correction contributes ~0.03 dex of this.
Use CIGALE for SF/Q separation; FSF stellar masses are consistent after
applying the same bin offset.

Adopted stellar mass thresholds (matched to Mr samples above):

| Mr cut | CIGALE log M* | FSF log M* | zmax |
|--------|---------------|------------|------|
| −21.5  | > 11.10       | > 11.45    | 0.42 |
| −20.5  | > 10.60       | > 10.95    | 0.34 |
| −19.5  | > 10.10       | > 10.45    | 0.27 |
| −18.5  | >  9.50       | >  9.85    | 0.21 |
| −17.5  | >  8.90       | >  9.25    | 0.17 |

SF/Q separation: CIGALE log(sSFR/yr⁻¹) = −11 (LOGSFR − LOGM).
Do not use FSF for SF/Q separation — FSF has a floor near −11 for quiescent
galaxies and cannot cleanly separate the populations.

Provenance flags (cigflag / fsfflag): 1 = direct TARGETID match,
2 = first-pass KD-tree donor, 3 = second-pass nearest-redshift donor,
−1 = unresolved (NaN values).

## Computing environment

All work runs on NERSC Perlmutter. Login nodes are sufficient for most steps;
compute nodes are required for `make_checkpoint.py`, `make_checkpoint_snap.py`
and all clustering scripts. **Always request 512 GB** — 128 GB causes OOM
failures on large random files (50M rows).

```bash
# Allocate an interactive compute node (max 4 hr on interactive QOS)
salloc --nodes=1 --qos=interactive --constraint=cpu --account=desi \
       --mem=512GB --time=04:00:00

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
The Uchuu simulation uses Om0=0.3089 (Planck 2015); this is used only in
`make_checkpoint.py` for the real-space to redshift mapping of subhaloes.

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sig_lgv_ref` | 0.18 | log₁₀(Vpeak) scatter amplitude σ_{v,0} at pivot |
| `sig_lgv_slope` | 0.30 | Gradient σ_{v,1} of scatter with log₁₀(Vpeak) |
| `lgv_pivot` | 2.6 | log₁₀(V_piv) pivot point |
| `f_colour` | 0.0 | Assembly-bias colour scatter parameter σ_col |
| `alpha` | 0.0 | Correlated scatter parameter α_v |
| `obs_seed` | 0 | Observer position seed (0 = box centre) |
| `min_Vpeak` | 70 km/s | Minimum subhalo Vpeak |
| `N_neigh_vpeak` | 1000 | Neighbours for FracZpeak estimate |
| `N_neigh_lum` | 50 | Neighbours for colour matching |
| `nzbins` | 200 | Number of redshift shells for SHAM |
| `Δz` | 0.03 | Redshift window for random catalogue cloning |

## Typical workflow

**Always run scripts from `~/DirectSHAM/pipeline/`** — a `data/` symlink there points to the K-correction files needed by `MockCode.py`. Running from the parent directory will fail with a `FileNotFoundError`.

```bash
cd ~/DirectSHAM
REF=0.18; SLOPE=0.30; FC=0.00; AL=0.00; SEED=0

# 1. Build the random catalogue (login node, once per data release)
python pipeline/make_random.py

# 2. Build the subhalo checkpoint (compute node, ~1 hr)
python pipeline/make_checkpoint.py --reg NS --obs_seed $SEED \
    --sig_lgv_ref $REF --sig_lgv_slope $SLOPE

# 3. Run SHAM to produce mock (login node)
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

# 9. Measure stellar mass clustering (compute node)
bash pipeline/run_sfq_clustering.sh

# 10. Measure LSS large-scale clustering (compute node)
#     Data:
python pipeline/measure_lss_clustering.py --mode data
#     Lightcone mock:
python pipeline/measure_lss_clustering.py --mode mock
#     Cut-sky mock:
python pipeline/measure_lss_clustering.py --mode mock --cutsky \
    --sig_lgv_ref $REF --sig_lgv_slope $SLOPE
```

## Data locations (NERSC)

| Item | Path |
|------|------|
| BGS DR2 data | `$DATA_DIR/6thOct/BGS_BRIGHT_6thOct.fits` |
| BGS augmented (with M*) | `$DATA_DIR/6thOct/BGS_BRIGHT_6thOct_augmented.fits` |
| Fordata randoms | `$DATA_DIR/6thOct/rancat_fordata_{N,S,NGC,SGC}.fits` |
| Formock randoms | `$DATA_DIR/6thOct/rancat_formock_{N,S,NGC,SGC}.fits` |
| Mock catalogues | `$DATA_DIR/6thOct/N_neigh_lum50/seed000_sig_lgv0.18_0.30_fc0.00_al0.00/mock_{N,S,NGC,SGC}.fits` |
| Lightcone catalogues | `$DATA_DIR/6thOct/lightcone_sig_lgv0.18_0.30_fc0.00_al0.00/mock_lightcone_{N,S,NGC,SGC}.fits` |
| Lightcone checkpoints | `$DATA_DIR/6thOct/lightcone_sig_lgv0.18_0.30_fc0.00_al0.00/subhalos_{N,S}_snap{40,41,43,45,47,50}.fits` |
| Clustering output (cut-sky) | `$DATA_DIR/6thOct/.../clustering/` |
| Clustering output (lightcone) | `$DATA_DIR/6thOct/lightcone_.../clustering/` |
| LSS clustering output | `...clustering/lss/{data,mock,mock_cutsky}/smu/` |
| CIGALE VAC (v2.0) | `/global/cfs/cdirs/desi/science/gqp/y3gqp/v2.0/st_mass_MS_cigale/BGS_BRIGHT-loa-LSSv2.1-cigale-v1.0.fits` |
| FSF VAC | `/global/cfs/cdirs/desicollab/science/gqp/y3gqp/v1.0/parent/BGS_ANY-y3gqp-parent-fsf-v1.0.fits` |
| Parent catalogues (PIP weights) | `/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1/` |
| Uchuu halo catalogues | `/global/cfs/cdirs/desi/mocks/cai/Uchuu-SHAM/Uchuu-halo-catalogs/halodir_{index:03d}/` |
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
- Moustakas J. et al., 2023 (FastSpecFit; ascl:2308.005)
- Siudek M. et al., 2024, A&A, 691, A308 (CIGALE for DESI BGS)
- Smith A. et al., 2022, MNRAS, 516, 1062
- Kitzbichler M. G. & White S. D. M., 2007, MNRAS, 376, 2
