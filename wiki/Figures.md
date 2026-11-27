# Figures

Script-to-figure mapping for Cole et al. (2026). All plotting scripts are in
`pipeline/`. Clustering inputs are `.npy` files loadable with
`TwoPointCorrelationFunction.load()` from pycorr.

## Figure inventory

| Fig. | Script | Description |
|------|--------|-------------|
| 1 | — | Schematic of volume-limited samples for abundance matching: bluest/reddest/global magnitude limits vs redshift |
| 2 | — | Sky coverage of North and South mock catalogues |
| 3 | — | Redshift distributions: BGS data, randoms, mock (North upper, South lower) |
| 4 | `plot_nz_mstar.py` | Comoving number density n(z) for Mr threshold samples: data, randoms, mock |
| 5 | — | Redshift cone plots: BGS data and Uchuu mock (180° < RA < 275°, 30° < Dec < 35°) |
| 6 | — | Absolute magnitude distributions in redshift shells: data, randoms, mock |
| 7 | — | Colour distributions in redshift shells: data, randoms, mock |
| 8 | `S5PanelClustering1.py` | ξ₀ and ξ₂ for four constant-σ_v models vs data, five luminosity bins |
| 9 | `S5PanelClustering2.py` | ξ₀ and ξ₂ for tuned luminosity-dependent scatter model (σ_{v,0}=0.18, σ_{v,1}=0.30, α_v=0.02) |
| 10 | `compute_hod.py` | Mean halo occupation distribution ⟨N⟩ vs Mhost for each Mr threshold sample |
| 11 | `S5PanelColourClustering2.py` | Red/blue ξ₀ and ξ₂: data and mock, five luminosity bins (Mr<-21.5 to -17.5) |
| 12 | `compute_bias.py` | Linear bias b vs Mr threshold for full, red and blue samples (data circles, mock squares) |
| 13 | `inspect_stellar_masses.py` | CIGALE vs FSF stellar mass comparison: 2D histogram, median, 5th–95th percentile |
| 14 | `plot_nz_mstar.py` | n(z) for Mr and M* threshold samples: CIGALE and FSF curves |
| 15 | `compare_clustering_mstar.py` | ξ₀ and ξ₂ for CIGALE (red) and FSF (green) stellar mass threshold samples |
| 16 | `compare_clustering_sfq.py` | ξ₀ and ξ₂ for star-forming and quenched subsamples, four stellar mass bins (CIGALE sSFR) |
| 17 | `S5PanelClustering2.py` (lightcone mode) | Lightcone mock ξ₀ and ξ₂ vs data, five luminosity bins |
| 18 | `plot_lss_clustering.py` | LSS sample ξ₀ and ξ₂: data (black/grey), lightcone mock (red), cut-sky mock (blue) |

### Regeneration status for the 16thNov/al0.02 model

| Fig. | Status |
|------|--------|
| 12 | Updated |
| 10 | Pending — rerun `compute_hod.py` on 16thNov mock |
| 16 | Pending — SF/passive clustering for 16thNov mock (`run_sfq_clustering.sh`) |
| 17 | Pending — lightcone rebuild for 16thNov/al0.02 |
| 18 | Pending — `measure_lss_clustering.py` on 16thNov lightcone |

Figures 1, 2, 3, 5, 6, 7 are produced by inspection/diagnostic scripts or
notebooks and do not have a dedicated standalone plotting script in
`pipeline/`.

## Notes on individual figures

### Fig. 8 — constant scatter comparison
`S5PanelClustering1.py` reads clustering output for four constant σ_v values.
Data points use PIP weights; mock curves use unit weights. Key result: no
single constant σ_v fits both bright and faint samples simultaneously.

### Fig. 9 — tuned model
`S5PanelClustering2.py` reads from the best-fit model directory
`seed000_sig_lgv0.18_0.30_fc0.00_al0.02`. Hardwired paths — no CLI args needed.
Reproduces data well across all five luminosity bins.

### Fig. 10 — HOD
`compute_hod.py` reads mock catalogues directly (no clustering .npy files needed)
and computes mean ⟨N⟩ for centrals and satellites in Mhost bins.

### Fig. 11 — colour-dependent clustering
`S5PanelColourClustering2.py` applies the red/blue colour split
(⁰·¹(g−r) ≷ 0.21 − 0.03 Mr) to both data and mock. Hardwired paths to
16thNov model with al0.02. `compare_clustering_colour.py` is a more
flexible CLI-driven version for diagnostic use.

### Fig. 12 — linear bias
`compute_bias.py` fits the Kaiser monopole model over 10 < s < 40 h⁻¹ Mpc.
Colour split in `get_median_redshift()` and `get_halo_properties()` now uses
`0.21 - 0.03*ABSMAG_RP1` (previously hardcoded `gmr >= 0.6`; fixed Nov 17 2026).
**Updated** for the 16thNov/al0.02 model.

### Fig. 15 — stellar mass clustering
`compare_clustering_mstar.py` reads mock from the 16thNov model directory
(3dp z formatting). <!-- CHECK: data still read from 6thOct (2dp), or moved to 16thNov? -->
CIGALE solid curves, FSF dotted curves.

### Fig. 16 — SF/quenched clustering
`compare_clustering_sfq.py` reads data and mock SF/passive clustering results.
Both data and mock directories use 2dp z formatting.

### Fig. 17 — lightcone clustering
Produced from lightcone mock clustering output under
`lightcone_sig_lgv0.18_0.30_fc0.00_al0.02/clustering/`. Script TBC — may
reuse `S5PanelClustering2.py` pointed at lightcone paths, or a dedicated
`plot_lightcone_clustering.py`.

### Fig. 18 — LSS large-scale clustering
`plot_lss_clustering.py` reads from `lightcone_.../clustering/lss/`.
Mock monopole tracks data well 1–60 Mpc/h including BAO peak.

## Clustering output file locations

| Sample type | Data | Mock (cut-sky) | Mock (lightcone) |
|-------------|------|----------------|-----------------|
| Luminosity threshold | `16thNov/clustering/data/` | `model_dir/clustering/` | `lightcone_dir/clustering/` |
| Colour split (red/blue) | `16thNov/clustering/data/Mr*_z*_{red,blue}/` | `model_dir/clustering/Mr*_z*_{red,blue}/` | — |
| Stellar mass / SF/Q | `16thNov/clustering/data/` | `model_dir/clustering/` | — |
| LSS large-scale | `lightcone_dir/clustering/lss/data/` | `model_dir/clustering/lss/mock_cutsky/` | `lightcone_dir/clustering/lss/mock/` |
| CIGALE KP bins w(r_p) | `16thNov/clustering/data/Mcig*/rppi/` | `model_dir/clustering/Mcig*/rppi/` | — |

Clustering results are saved as `.npy` files and can be reloaded with:
```python
from pycorr import TwoPointCorrelationFunction
result = TwoPointCorrelationFunction.load('allcounts_BGS_BRIGHT_GCcomb_....npy')
```

## Directory naming conventions

Data and mock directories use **2dp** z formatting: `Mr-21.5_z0.02-0.42_red`.
Mock stellar mass directories use **3dp** z formatting: `Mcig11.1_z0.020-0.420`.
Scripts must use the correct format when constructing paths — a mismatch
causes silent failures (file not found, no data plotted).
