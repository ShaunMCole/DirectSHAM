"""
pipeline_utils.py — Shared utilities for the DirectSHAM pipeline.

Provides:
  - make_paths()     : construct all standard file paths for a given set of
                       pipeline parameters, so no script ever builds a path
                       by hand.
  - write_fits_meta(): attach standard pipeline metadata to a FITS header
                       before writing, so every output file is self-describing.
  - check_fits_meta(): verify that metadata stored in a file matches the
                       parameters being used, catching accidental mismatches
                       early.
  - setup_logging()  : configure a standard logger that writes to both the
                       terminal and a dated log file under logs/.

All functions import from config.py only; they do not import catalogue_analysis
or MockCode so that they can be used from any script without loading the full
DESI environment.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import (
    DATA_DIR,
    DATA_DATE,
    DATA_DATE_DIR,
    MIN_VPEAK,
    N_NEIGH_VPEAK,
    N_NEIGH_LUM,
    DEFAULT_F_COLOUR,
    DEFAULT_ALPHA,
    NLUM_DIR_TEMPLATE,
    MODEL_DIR_TEMPLATE,
    RANCAT_FORDATA_TEMPLATE,
    RANCAT_FORDATA_NS_TEMPLATE,
    RANCAT_FORMOCK_TEMPLATE,
    RANCAT_FORMOCK_NGC_TEMPLATE,
    RANCAT_FORMOCK_SGC_TEMPLATE,
    SUBHALO_FITS_TEMPLATE,
    MOCK_TEMPLATE,
    MOCK_NGC_TEMPLATE,
    MOCK_SGC_TEMPLATE,
    CLUSTERING_TEMPLATE,
)


# =============================================================================
# Path construction
# =============================================================================

def make_paths(
    reg=None,
    obs_seed=None,
    sig_lgv_ref=None,
    sig_lgv_slope=None,
    f_colour=None,
    alpha=None,
    cap=None,
    mr_cut=None,
    zmin=None,
    zmax=None,
    min_vpeak=None,
    n_neigh_vpeak=None,
    n_neigh_lum=None,
):
    """
    Construct all standard pipeline file paths for the given parameters.

    All date-specific files live under DATA_DATE_DIR (= DATA_DIR / DATA_DATE).
    All model-specific files live under a model subdirectory of DATA_DATE_DIR,
    named seed{obs_seed:03d}_sig_lgv{sig_lgv_ref:.2f}_{sig_lgv_slope:.2f}.

    Only the parameters relevant to the paths you need must be supplied;
    the rest can be left as None.  Passing an unneeded parameter does no
    harm.

    Parameters
    ----------
    reg : str, optional
        Photometric region: 'N' or 'S'.
    obs_seed : int, optional
        Observer position seed (0 = box centre).
    sig_lgv_ref : float, optional
        Vpeak scatter at the pivot log10(Vpeak).
    sig_lgv_slope : float, optional
        Gradient of scatter with log10(Vpeak).
    cap : str, optional
        Galactic cap: 'NGC' or 'SGC'.
    mr_cut : float, optional
        Absolute magnitude cut (e.g. -21.0).
    zmin : float, optional
        Minimum redshift of the clustering sample.
    zmax : float, optional
        Maximum redshift of the clustering sample.
    min_vpeak : float, optional
        Minimum Vpeak threshold. Defaults to MIN_VPEAK from config.
    n_neigh_vpeak : int, optional
        N_neigh_vpeak. Defaults to N_NEIGH_VPEAK from config.

    Returns
    -------
    dict
        Keys:
        model_dir             — model subdirectory (Path)
        fordata_ran_{reg}     — per-region fordata random catalogue
        fordata_ran           — combined N+S fordata random catalogue
        formock_ran_{reg}     — per-region formock random catalogue
        formock_ran_NGC       — NGC formock random catalogue
        formock_ran_SGC       — SGC formock random catalogue
        subhalo_checkpoint    — Uchuu checkpoint file
        mock_{reg}            — per-region mock galaxy catalogue
        mock_NGC              — NGC mock galaxy catalogue
        mock_SGC              — SGC mock galaxy catalogue
        clustering            — pycorr clustering output (.npy)

    Examples
    --------
    # Paths needed to make a North checkpoint:
    paths = make_paths(reg='N', obs_seed=0, sig_lgv_ref=0.18, sig_lgv_slope=0.30)
    print(paths['subhalo_checkpoint'])

    # Paths needed to measure clustering:
    paths = make_paths(obs_seed=0, sig_lgv_ref=0.18, sig_lgv_slope=0.30,
                       cap='NGC', mr_cut=-21.0, zmin=0.02, zmax=0.42)
    print(paths['clustering'])
    """
    mvp   = min_vpeak    if min_vpeak    is not None else MIN_VPEAK
    nnvp  = n_neigh_vpeak if n_neigh_vpeak is not None else N_NEIGH_VPEAK
    nnlum = n_neigh_lum  if n_neigh_lum  is not None else N_NEIGH_LUM
    fc    = f_colour     if f_colour     is not None else DEFAULT_F_COLOUR
    al    = alpha        if alpha        is not None else DEFAULT_ALPHA

    paths = {}

    # --- fordata randoms (shared; live in DATA_DATE_DIR) ---
    if reg is not None:
        paths[f"fordata_ran_{reg}"] = DATA_DATE_DIR / RANCAT_FORDATA_TEMPLATE.format(
            reg=reg
        )
    paths["fordata_ran"] = DATA_DATE_DIR / RANCAT_FORDATA_NS_TEMPLATE

    # --- formock randoms (shared; live in DATA_DATE_DIR) ---
    if reg is not None:
        paths[f"formock_ran_{reg}"] = DATA_DATE_DIR / RANCAT_FORMOCK_TEMPLATE.format(
            reg=reg
        )
    paths["formock_ran_NGC"] = DATA_DATE_DIR / RANCAT_FORMOCK_NGC_TEMPLATE
    paths["formock_ran_SGC"] = DATA_DATE_DIR / RANCAT_FORMOCK_SGC_TEMPLATE

    # --- N_neigh_lum and model directories ---
    if obs_seed is not None and sig_lgv_ref is not None:
        nlum_dir  = DATA_DATE_DIR / NLUM_DIR_TEMPLATE.format(n_neigh_lum=nnlum)
        model_dir = nlum_dir / MODEL_DIR_TEMPLATE.format(
            obs_seed=obs_seed,
            sig_lgv_ref=sig_lgv_ref,
            sig_lgv_slope=sig_lgv_slope,
            f_colour=fc,
            alpha=al,
        )
        paths["nlum_dir"]   = nlum_dir
        paths["model_dir"]  = model_dir

        # --- Uchuu checkpoint (inside model dir) ---
        if reg is not None:
            paths["subhalo_checkpoint"] = model_dir / SUBHALO_FITS_TEMPLATE.format(
                reg=reg,
                min_vpeak=mvp,
                n_neigh_vpeak=nnvp,
            )

        # --- mock catalogues (inside model dir) ---
        if reg is not None:
            paths[f"mock_{reg}"] = model_dir / MOCK_TEMPLATE.format(reg=reg)
        paths["mock_NGC"] = model_dir / MOCK_NGC_TEMPLATE
        paths["mock_SGC"] = model_dir / MOCK_SGC_TEMPLATE

        # --- clustering output (inside model dir under clustering/) ---
        if (cap is not None and mr_cut is not None
                and zmin is not None and zmax is not None):
            paths["clustering"] = (
                model_dir / "clustering"
                / CLUSTERING_TEMPLATE.format(
                    cap=cap,
                    mr_cut=mr_cut,
                    zmin=zmin,
                    zmax=zmax,
                )
            )

    return paths



from pathlib import Path
from config import (
    DATA_DIR, DATA_DATE, DATA_DATE_DIR,
    LIGHTCONE_DIR_TEMPLATE,
    LC_SUBHALO_TEMPLATE,
    LC_SHELL_MOCK_TEMPLATE,
    LC_MOCK_TEMPLATE,
    LC_MOCK_NGC_TEMPLATE,
    LC_MOCK_SGC_TEMPLATE,
)


def make_lightcone_paths(
    sig_lgv_ref,
    sig_lgv_slope,
    f_colour=0.0,
    alpha=0.0,
    snapshot=None,
    reg=None,
):
    """
    Return a dict of Path objects for the lightcone pipeline.

    Parameters
    ----------
    sig_lgv_ref, sig_lgv_slope, f_colour, alpha : floats
        Model parameters (same as make_paths).
    snapshot : int or None
        Snapshot index (e.g. 45).  Required for per-snapshot paths;
        if None those entries are omitted.
    reg : str or None
        Region ('N' or 'S').  Required for per-region paths;
        if None those entries are omitted.

    Returns
    -------
    dict with keys:
        lightcone_dir          - Path to the lightcone model directory
        subhalo_fits           - checkpoint file (needs snapshot + reg)
        shell_mock             - per-snapshot full-range mock (needs snapshot + reg)
        mock_lightcone_N/S     - assembled lightcone catalogue
        mock_lightcone_NGC/SGC - NGC/SGC split catalogues
        clustering_dir         - clustering output directory
    """
    lc_dir_name = LIGHTCONE_DIR_TEMPLATE.format(
        sig_lgv_ref=sig_lgv_ref,
        sig_lgv_slope=sig_lgv_slope,
        f_colour=f_colour,
        alpha=alpha,
    )
    lightcone_dir = DATA_DATE_DIR / lc_dir_name

    paths = dict(lightcone_dir=lightcone_dir)

    # Per-snapshot, per-region paths
    if snapshot is not None and reg is not None:
        paths['subhalo_fits'] = lightcone_dir / LC_SUBHALO_TEMPLATE.format(
            reg=reg, snap=snapshot
        )
        paths['shell_mock'] = lightcone_dir / LC_SHELL_MOCK_TEMPLATE.format(
            reg=reg, snap=snapshot
        )

    # Assembled lightcone catalogues (no snapshot needed)
    if reg is not None:
        paths['mock_lightcone'] = lightcone_dir / LC_MOCK_TEMPLATE.format(reg=reg)

    paths['mock_lightcone_NGC'] = lightcone_dir / LC_MOCK_NGC_TEMPLATE
    paths['mock_lightcone_SGC'] = lightcone_dir / LC_MOCK_SGC_TEMPLATE

    # Clustering output directory
    paths['clustering_dir'] = lightcone_dir / 'clustering'

    return paths

# =============================================================================
# FITS metadata helpers
# =============================================================================

# Standard FITS header keywords written by every pipeline stage.
# The value is a (value, comment) tuple.
_META_KEYS = {
    "PIPE_VER": ("DirectSHAM-1.0", "Pipeline version"),
    "CREATED":  (None,             "UTC datetime this file was written"),
}


def write_fits_meta(header, params):
    """
    Write standard pipeline metadata into a FITS header.

    Call this just before writing any output file.  The header will
    contain both the standard keys (_META_KEYS) and any key/value pairs
    supplied in params.

    Parameters
    ----------
    header : astropy.io.fits.Header  or  dict-like
        The header object to update in place.  Astropy Table.meta dicts
        are accepted as well as proper FITS Header objects.
    params : dict
        Pipeline parameters to record.  Recommended keys:

        STAGE     — which pipeline script wrote this file
        REG       — photometric region ('N', 'S', or absent for combined)
        OBS_SEED  — observer position seed
        SIG_LGV   — Vpeak scatter in dex
        MIN_VPEAK — Vpeak floor in km/s
        NNVPEAK   — N_neigh_vpeak
        NZBINS    — number of redshift bins
        DELTAZ    — DeltaZ window used when making randoms
        NMULT     — random oversampling factor
        DATADATE  — date tag of the input data catalogue

    Returns
    -------
    None  (header is modified in place)
    """
    header["PIPE_VER"] = ("DirectSHAM-1.0", "Pipeline version")
    header["CREATED"]  = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "UTC datetime this file was written",
    )
    for key, value in params.items():
        header[key] = value


def check_fits_meta(header, params, strict=True):
    """
    Verify that metadata stored in a FITS header matches expected values.

    Parameters
    ----------
    header : dict-like
        Header or Table.meta read from the file to be checked.
    params : dict
        Expected key/value pairs.  Only keys present in both header and
        params are checked; extra keys in either are ignored.
    strict : bool
        If True (default), raise ValueError on any mismatch.
        If False, log a warning instead.

    Returns
    -------
    bool
        True if all checked values match, False otherwise (only when
        strict=False; otherwise raises on mismatch).

    Raises
    ------
    ValueError
        If strict=True and any value does not match.
    """
    log = logging.getLogger(__name__)
    ok  = True

    for key, expected in params.items():
        if key not in header:
            continue  # key not stored — nothing to check
        found = header[key]
        if found != expected:
            msg = (
                f"Metadata mismatch for '{key}': "
                f"file has {found!r}, expected {expected!r}"
            )
            if strict:
                raise ValueError(msg)
            else:
                log.warning(msg)
                ok = False

    return ok


# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(script_name, log_dir=None, level=logging.INFO):
    """
    Configure a standard logger that writes to stdout and a dated log file.

    Parameters
    ----------
    script_name : str
        Short name of the calling script, used in the log filename and
        as the logger name (e.g. 'make_checkpoint', 'make_mock').
    log_dir : Path or str, optional
        Directory in which to write the log file.  Defaults to
        REPO_DIR / 'logs'.
    level : int, optional
        Logging level.  Default is logging.INFO.

    Returns
    -------
    logging.Logger
        Configured logger.  Use log.info(), log.warning(), log.error()
        throughout the calling script.

    Notes
    -----
    The log file is named  <script_name>_YYYYMMDD_HHMMSS.log  so that
    repeated runs do not overwrite previous logs.
    """
    from config import REPO_DIR  # imported here to avoid circular issues

    if log_dir is None:
        log_dir = REPO_DIR / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"{script_name}_{timestamp}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler   = logging.FileHandler(log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)
    for h in (file_handler, stdout_handler):
        h.setFormatter(fmt)

    log = logging.getLogger(script_name)
    log.setLevel(level)
    log.addHandler(file_handler)
    log.addHandler(stdout_handler)

    log.info(f"Log file: {log_file}")
    return log
