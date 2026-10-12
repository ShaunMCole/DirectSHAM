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
    MIN_VPEAK,
    N_NEIGH_VPEAK,
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
    sig_lgv=None,
    cap=None,
    mr_cut=None,
    zmin=None,
    zmax=None,
    date=None,
    min_vpeak=None,
    n_neigh_vpeak=None,
):
    """
    Construct all standard pipeline file paths for the given parameters.

    Only the parameters relevant to the paths you need must be supplied;
    the rest can be left as None.  Passing an unneeded parameter does no
    harm.  Attempting to access a path whose required parameters were not
    supplied will raise a clear KeyError.

    Parameters
    ----------
    reg : str, optional
        Photometric region: 'N' or 'S'.
    obs_seed : int, optional
        Observer position seed (0 = box centre).
    sig_lgv : float, optional
        log10(Vpeak) scatter in dex.
    cap : str, optional
        Galactic cap: 'NGC' or 'SGC'.
    mr_cut : float, optional
        Absolute magnitude cut applied to the clustering sample (e.g. -21.0).
    zmin : float, optional
        Minimum redshift of the clustering sample.
    zmax : float, optional
        Maximum redshift of the clustering sample.
    date : str, optional
        Date tag for catalogue filenames.  Defaults to DATA_DATE from config.
    min_vpeak : float, optional
        Minimum Vpeak threshold used when making the checkpoint file.
        Defaults to MIN_VPEAK from config.
    n_neigh_vpeak : int, optional
        Number of Vpeak neighbours used when computing FracZpeak.
        Defaults to N_NEIGH_VPEAK from config.

    Returns
    -------
    dict
        Dictionary mapping path name strings to pathlib.Path objects.
        Keys are:

        fordata_ran_{reg}     — per-region fordata random catalogue
        fordata_ran           — combined N+S fordata random catalogue
        formock_ran_{reg}     — per-region formock random catalogue
        formock_ran_NGC       — NGC formock random catalogue
        formock_ran_SGC       — SGC formock random catalogue
        subhalo_checkpoint    — Uchuu checkpoint file (expensive step)
        mock_{reg}            — per-region mock galaxy catalogue
        mock_NGC              — NGC mock galaxy catalogue
        mock_SGC              — SGC mock galaxy catalogue
        clustering            — pycorr clustering output (.npy)

    Examples
    --------
    # Paths needed to make a North checkpoint:
    paths = make_paths(reg='N', obs_seed=0, sig_lgv=0.10)
    print(paths['subhalo_checkpoint'])

    # Paths needed to measure clustering:
    paths = make_paths(obs_seed=0, sig_lgv=0.10, cap='NGC',
                       mr_cut=-21.0, zmin=0.1, zmax=0.4)
    print(paths['clustering'])
    """
    d      = date         if date         is not None else DATA_DATE
    mvp    = min_vpeak    if min_vpeak    is not None else MIN_VPEAK
    nnvp   = n_neigh_vpeak if n_neigh_vpeak is not None else N_NEIGH_VPEAK

    paths = {}

    # --- fordata randoms ---
    if reg is not None:
        paths[f"fordata_ran_{reg}"] = DATA_DIR / RANCAT_FORDATA_TEMPLATE.format(
            date=d, reg=reg
        )
    paths["fordata_ran"] = DATA_DIR / RANCAT_FORDATA_NS_TEMPLATE.format(date=d)

    # --- formock randoms ---
    if reg is not None:
        paths[f"formock_ran_{reg}"] = DATA_DIR / RANCAT_FORMOCK_TEMPLATE.format(
            date=d, reg=reg
        )
    paths["formock_ran_NGC"] = DATA_DIR / RANCAT_FORMOCK_NGC_TEMPLATE.format(date=d)
    paths["formock_ran_SGC"] = DATA_DIR / RANCAT_FORMOCK_SGC_TEMPLATE.format(date=d)

    # --- Uchuu checkpoint ---
    if reg is not None and obs_seed is not None and sig_lgv is not None:
        paths["subhalo_checkpoint"] = DATA_DIR / SUBHALO_FITS_TEMPLATE.format(
            reg=reg,
            min_vpeak=mvp,
            obs_seed=obs_seed,
            sig_lgv=sig_lgv,
            n_neigh_vpeak=nnvp,
        )

    # --- mock catalogues ---
    if reg is not None and obs_seed is not None and sig_lgv is not None:
        paths[f"mock_{reg}"] = DATA_DIR / MOCK_TEMPLATE.format(
            reg=reg, obs_seed=obs_seed, sig_lgv=sig_lgv
        )
    if obs_seed is not None and sig_lgv is not None:
        paths["mock_NGC"] = DATA_DIR / MOCK_NGC_TEMPLATE.format(
            obs_seed=obs_seed, sig_lgv=sig_lgv
        )
        paths["mock_SGC"] = DATA_DIR / MOCK_SGC_TEMPLATE.format(
            obs_seed=obs_seed, sig_lgv=sig_lgv
        )

    # --- clustering output ---
    if (cap is not None and obs_seed is not None and sig_lgv is not None
            and mr_cut is not None and zmin is not None and zmax is not None):
        paths["clustering"] = DATA_DIR / CLUSTERING_TEMPLATE.format(
            cap=cap,
            obs_seed=obs_seed,
            sig_lgv=sig_lgv,
            mr_cut=mr_cut,
            zmin=zmin,
            zmax=zmax,
        )

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
