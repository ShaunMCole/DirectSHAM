"""
config.py — Central configuration for the DirectSHAM pipeline.

All hardcoded paths, survey parameters, simulation parameters and
default algorithmic settings live here. Every other pipeline script
imports from this module rather than defining its own constants.

To adapt the pipeline to a new data release or a different machine,
edit this file only.
"""

from pathlib import Path
from astropy.cosmology import FlatLambdaCDM

# =============================================================================
# Cosmology
# =============================================================================

# Standard DESI analysis cosmology, matching cosmology.py which is used by
# catalogue_analysis.py and MockCode.py throughout the pipeline.
# All comoving distances are in Mpc/h (H0=100).
cosmo = FlatLambdaCDM(H0=100, Om0=0.313, Tcmb0=2.725)

# The Uchuu simulation was run with a slightly different cosmology (Om0=0.3089,
# consistent with Planck 2015).  This value is used ONLY in make_checkpoint.py
# when converting Uchuu comoving distances to redshifts.  Everywhere else the
# analysis cosmology above is used, including when pycorr converts mock
# redshifts back to distances for the clustering measurement.  The resulting
# sub-percent inconsistency in the mock redshifts is negligible.
cosmo_uchuu = FlatLambdaCDM(H0=100, Om0=0.3089, Tcmb0=2.725)

# =============================================================================
# Top-level directory layout
# =============================================================================

# Root of the pipeline source tree (this file's parent directory's parent).
PIPELINE_DIR = Path(__file__).resolve().parent
REPO_DIR     = PIPELINE_DIR.parent

# All large output files (catalogues, checkpoints, clustering results) live
# under DATA_DIR.  This is placed under the shared DESI CFS allocation so
# that collaborators (e.g. efdez at IAA) can read and write here without
# needing access to smcole's home directory.
# Permissions should be set with:  chmod -R g+rX <DATA_DIR>
# and the directory should belong to the desi group (gid=58102).
DATA_DIR = Path("/global/cfs/cdirs/desi/users/smcole/DirectSHAM")

# =============================================================================
# Input data files
# =============================================================================

# Date/version tags used throughout the pipeline.  Changing these here
# updates all downstream filenames automatically.
DATA_DATE     = "6thOct"
LSS_VERSION   = "v2.1"
DATA_DATE_DIR = DATA_DIR / DATA_DATE   # root for all date-specific files

# Processed BGS catalogue produced by Augment_BGS_cat.ipynb upstream.
# Date is kept in the filename so the file is self-describing when shared.
BGS_CAT = DATA_DATE_DIR / f"BGS_BRIGHT_{DATA_DATE}.fits"

# Official DESI parent catalogues (read-only; not produced by this pipeline).
DESI_LSS_BASE = Path(
    f"/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/{LSS_VERSION}/PIP/"
)
# Format with i=0,1,2,... to get successive random catalogue files.
DESI_RANDOMS_TEMPLATE = str(DESI_LSS_BASE / "BGS_BRIGHT_{i}_clustering.ran.fits")

# Official DESI tiles file used for the survey mask.
TILES_FILE = Path(
    "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/tiles-BRIGHT.fits"
)

# Uchuu halo catalogue directory template (filled in with snapshot index).
UCHUU_HALODIR_TEMPLATE = (
    "/global/cfs/cdirs/desi/mocks/cai/Uchuu-SHAM/"
    "Uchuu-halo-catalogs/halodir_{index:03d}"
)

# =============================================================================
# Survey selection parameters  (per photometric region)
# =============================================================================
# Values taken directly from catalogue_analysis.selection().
# If that function is updated, update these to match.

# Sky areas in units of fraction of the full sphere (as used in ca.selection).
# Values are from Y3 (DR2) catalogues:
#   North: 3827.50 sq deg,  South: 8527.58 sq deg
_AREA_N = 3827.50 / (4 * 3.141592653589793 * (180.0 / 3.141592653589793)**2)
_AREA_S = 8527.58 / (4 * 3.141592653589793 * (180.0 / 3.141592653589793)**2)

SELECTION = {
    "N": {
        "faint":  19.54,   # BGS apparent-magnitude faint limit in the North
        "bright":  0.0,    # BGS apparent-magnitude bright limit
        "zmin":    0.01,
        "zmax":    0.50,
        "area":   _AREA_N, # sky area as fraction of full sphere
        "Qevol":   0.78,   # luminosity evolution parameter (same as Qr)
        "Qr":      0.78,   # r-band evolution correction coefficient
        "Qg":      0.91,   # g-band evolution correction coefficient
        "Qz":      0.45,   # z-band evolution correction coefficient
        "Qw1":     0.70,   # W1-band evolution correction coefficient
    },
    "S": {
        "faint":  19.50,   # BGS apparent-magnitude faint limit in the South
        "bright":  0.0,
        "zmin":    0.01,
        "zmax":    0.50,
        "area":   _AREA_S,
        "Qevol":   0.78,
        "Qr":      0.78,
        "Qg":      0.91,
        "Qz":      0.45,
        "Qw1":     0.70,
    },
}

# Sky mask boundaries that define North vs South within the DESI footprint.
# North: RA in [85, 305] AND DEC > 32.375 (AND inside official mask).
# South: inside official mask AND NOT North.
NORTH_RA_MIN  =  85.0
NORTH_RA_MAX  = 305.0
NORTH_DEC_MIN =  32.375

# =============================================================================
# Uchuu simulation parameters
# =============================================================================

LBOX           = 2000.0   # Simulation box side length [Mpc/h]
FRAC_BOX       = 0.8      # Sphere radius = FRAC_BOX * LBOX
SNAPSHOT_INDEX = 45       # Snapshot used for cut-sky mocks (z ≈ 0.19)

# Columns to read from the Uchuu hdf5 files.
UCHUU_COLUMNS = (
    "id", "pid", "Mvir_all", "Vpeak",
    "x", "y", "z", "vx", "vy", "vz",
    "Mpeak_Scale", "Rvir", "rs",
)

# Minimum log10(Mvir_all / [Msun/h]) when reading Uchuu files.
MIN_LOG_MVIR = 9.0

# =============================================================================
# Random catalogue parameters
# =============================================================================

DELTA_Z  = 0.03   # Maximum redshift displacement when cloning a galaxy
NMULT    = 7      # Target oversampling factor of randoms relative to data
NSAMP    = 10     # Mean clones per galaxy per iteration in random_catalogue()
N_ITER   = 5      # Number of overdensity iterations in random_catalogue()

# Random seed used when downsampling the South randoms to match North NMULT.
NMULT_DOWNSAMPLE_SEED = 12345

# =============================================================================
# Mock catalogue parameters (defaults; overridden on the command line)
# =============================================================================

# Vpeak scatter parametrisation: sig_lgv = sig_lgv_ref + sig_lgv_slope * (log10(Vpeak) - 2.6)
# sig_lgv_ref: scatter at log10(Vpeak/km/s) = 2.6  (~Mr = -21)
# sig_lgv_slope: gradient with log10(Vpeak) (positive = more scatter for brighter/more massive haloes)
DEFAULT_SIG_LGV_REF   = 0.14
DEFAULT_SIG_LGV_SLOPE = 0.24
DEFAULT_LGV_PIVOT     = 2.6   # log10(Vpeak/[km/s]) pivot point
DEFAULT_OBS_SEED = 0       # Observer position seed (0 = box centre)
MIN_VPEAK        = 70.0    # Discard subhalos below this Vpeak [km/s]
N_NEIGH_VPEAK    = 1000    # Neighbours used to compute FracZpeak
N_NEIGH_LUM      = 50      # Neighbours in luminosity for colour matching
N_ZBINS          = 200     # Number of redshift slices for SHAM

# K-correction reference colour used for the global (colour-independent)
# absolute magnitude ABSMAG_R, following the GAMA/Moore+2026 convention.
KCORR_REF_GMR = 0.77

# =============================================================================
# Filename conventions
# =============================================================================
# All filename construction is centralised in pipeline_utils.py (see
# make_paths()), which uses the templates below.  Do not construct
# filenames ad-hoc in other scripts.

# Model directory name — all files for one scatter realisation live here,
# under DATA_DATE_DIR.
# e.g. DATA_DATE_DIR / "seed000_sig_lgv0.18_0.30/"
MODEL_DIR_TEMPLATE = "seed{obs_seed:03d}_sig_lgv{sig_lgv_ref:.2f}_{sig_lgv_slope:.2f}"

# fordata random catalogue — shared across all models; date captured by
# parent directory, so not repeated in filename.
RANCAT_FORDATA_TEMPLATE    = "rancat_fordata_{reg}.fits"
RANCAT_FORDATA_NS_TEMPLATE = "rancat_fordata.fits"        # N+S combined

# formock random catalogue — shared across all models.
RANCAT_FORMOCK_TEMPLATE     = "rancat_formock_{reg}.fits"
RANCAT_FORMOCK_NGC_TEMPLATE = "rancat_formock_NGC.fits"
RANCAT_FORMOCK_SGC_TEMPLATE = "rancat_formock_SGC.fits"

# Uchuu checkpoint file — lives inside the model directory.
# obs_seed and sig_lgv are encoded in the model directory name, not here.
SUBHALO_FITS_TEMPLATE = (
    "subhalos_{reg}_{min_vpeak:.1f}_N_neigh_vpeak{n_neigh_vpeak:04d}.fits"
)

# Mock galaxy catalogue — lives inside the model directory.
MOCK_TEMPLATE     = "mock_{reg}.fits"
MOCK_NGC_TEMPLATE = "mock_NGC.fits"
MOCK_SGC_TEMPLATE = "mock_SGC.fits"

# Clustering output — lives inside the model directory under clustering/.
# cap, mr_cut and redshift range distinguish samples within the model dir.
CLUSTERING_TEMPLATE = (
    "xi_{cap}_Mr{mr_cut:.1f}_z{zmin:.2f}-{zmax:.2f}.npy"
)
