"""
split_ngcsgc.py — Split mock and random catalogues into NGC and SGC subsets.

This script combines all the NGC/SGC splitting tasks from NGCSGC_split.ipynb
into a single command-line tool.  It can split any combination of:

  --mocks      : combine mock_N + mock_S, split into mock_NGC + mock_SGC
  --formock    : combine formock_N + formock_S randoms, split into NGC + SGC
  --fordata    : split the combined fordata random catalogue into NGC + SGC

All three are run by default if no flags are given.

Usage examples
--------------
# Run all three splits (typical use):
python split_ngcsgc.py --obs_seed 0 --sig_lgv 0.10

# Run only the mock split:
python split_ngcsgc.py --obs_seed 0 --sig_lgv 0.10 --mocks

# Run only the fordata random split (no mock parameters needed):
python split_ngcsgc.py --fordata
"""

import argparse
import sys
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import astropy.units as u
from astropy.io.fits.verify import VerifyWarning

# Pipeline modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_OBS_SEED, DEFAULT_SIG_LGV
from pipeline_utils import make_paths, write_fits_meta, check_fits_meta, setup_logging

warnings.filterwarnings("ignore", category=VerifyWarning)


# =============================================================================
# Helpers
# =============================================================================

def add_cap_column(table):
    """
    Add a 'CAP' column ('NGC' or 'SGC') based on galactic latitude.

    Parameters
    ----------
    table : astropy Table
        Must contain 'RA' and 'DEC' columns in degrees (ICRS).

    Returns
    -------
    table : astropy Table
        The same table with a 'CAP' column added in place.
    """
    c = SkyCoord(ra=table['RA'] * u.deg, dec=table['DEC'] * u.deg, frame='icrs')
    table['CAP'] = np.where(c.galactic.b > 0, 'NGC', 'SGC')
    return table


def split_and_write(table, col, val_ngc, val_sgc, path_ngc, path_sgc,
                    drop_col=True, extra_meta=None, log=None):
    """
    Split a table on a column value and write NGC and SGC subsets to disk.

    Parameters
    ----------
    table : astropy Table
    col : str
        Column name to split on (typically 'CAP').
    val_ngc, val_sgc : str
        Values identifying each subset.
    path_ngc, path_sgc : Path
        Output file paths.
    drop_col : bool
        If True, remove the split column from the output files.
    extra_meta : dict, optional
        Additional metadata to write into the output headers.
    log : logging.Logger, optional
    """
    def _write(subset, path, label):
        if drop_col and col in subset.colnames:
            del subset[col]
        if extra_meta:
            write_fits_meta(subset.meta, extra_meta)
        subset.write(path, overwrite=True)
        msg = f"Written {label}: {len(subset):,} rows → {path}"
        if log:
            log.info(msg)
        else:
            print(msg)

    ngc = table[table[col] == val_ngc].copy()
    sgc = table[table[col] == val_sgc].copy()
    _write(ngc, path_ngc, "NGC")
    _write(sgc, path_sgc, "SGC")


# =============================================================================
# Task functions
# =============================================================================

def split_mocks(paths, log):
    """Concatenate mock_N + mock_S and split into mock_NGC + mock_SGC."""
    log.info("=== Splitting mock catalogues into NGC / SGC ===")

    path_N = paths["mock_N"]
    path_S = paths["mock_S"]

    for p in (path_N, path_S):
        if not p.exists():
            log.error(f"Input file not found: {p}")
            sys.exit(1)

    log.info(f"Reading {path_N}")
    mockN = Table.read(path_N)
    log.info(f"Reading {path_S}")
    mockS = Table.read(path_S)

    log.info("Concatenating N + S")
    mock = vstack([mockN, mockS], metadata_conflicts='silent')
    del mockN, mockS

    log.info("Assigning NGC/SGC by galactic latitude")
    mock = add_cap_column(mock)

    split_and_write(
        mock, 'CAP', 'NGC', 'SGC',
        paths["mock_NGC"], paths["mock_SGC"],
        extra_meta={"STAGE": "split_ngcsgc"},
        log=log,
    )
    del mock


def split_formock_randoms(paths, log):
    """Concatenate formock_N + formock_S randoms and split into NGC / SGC."""
    log.info("=== Splitting formock randoms into NGC / SGC ===")

    path_N = paths["formock_ran_N"]
    path_S = paths["formock_ran_S"]

    for p in (path_N, path_S):
        if not p.exists():
            log.error(f"Input file not found: {p}")
            sys.exit(1)

    log.info(f"Reading {path_N}")
    ranN = Table.read(path_N)
    log.info(f"Reading {path_S}")
    ranS = Table.read(path_S)

    # Verify metadata agrees between N and S (NMULT must match).
    log.info("Verifying N/S metadata consistency")
    ignore_meta = {'REG', 'FSKY', 'CREATED', 'PIPE_VER', 'NMULT'}
    for key in set(ranN.meta) | set(ranS.meta):
        if key in ignore_meta:
            continue
        valN = ranN.meta.get(key)
        valS = ranS.meta.get(key)
        if valN != valS:
            raise ValueError(
                f"Incompatible metadata for '{key}': N={valN!r}, S={valS!r}"
            )

    # Check NMULT separately with a tolerance since N and S are matched
    # but may differ by a tiny floating point amount.
    nmult_N = ranN.meta.get('NMULT', None)
    nmult_S = ranS.meta.get('NMULT', None)
    if nmult_N is not None and nmult_S is not None:
        reldiff = abs(nmult_N - nmult_S) / nmult_N
        if reldiff > 0.002:
            raise ValueError(
                f"NMULT mismatch: N={nmult_N:.6f}, S={nmult_S:.6f} "
                f"(relative difference {reldiff:.4f} > 0.2%)"
            )
    log.info(f"NMULT N={nmult_N}  S={nmult_S}")

    log.info("Concatenating N + S")
    ran = vstack([ranN, ranS], metadata_conflicts='silent')
    del ranN, ranS

    log.info("Assigning NGC/SGC by galactic latitude")
    ran = add_cap_column(ran)

    # Add WEIGHT column (= 1/NMULT) before splitting.
    ran['WEIGHT'] = 1.0 / ran.meta['NMULT']

    split_and_write(
        ran, 'CAP', 'NGC', 'SGC',
        paths["formock_ran_NGC"], paths["formock_ran_SGC"],
        extra_meta={"STAGE": "split_ngcsgc"},
        log=log,
    )
    del ran


def split_fordata_randoms(paths, log):
    """Split the combined fordata random catalogue into NGC and SGC subsets."""
    log.info("=== Splitting fordata randoms into NGC / SGC ===")

    path_in = paths["fordata_ran"]
    if not path_in.exists():
        log.error(f"Input file not found: {path_in}")
        sys.exit(1)

    log.info(f"Reading {path_in}")
    ran = Table.read(path_in)

    # The combined fordata catalogue already has a CAP column added by
    # RandomCatalogue.ipynb when it concatenated N and S.  If it is absent
    # (e.g. the file was written before that step was added) we compute it.
    if 'CAP' not in ran.colnames:
        log.info("CAP column not found — computing from RA/DEC")
        ran = add_cap_column(ran)

    path_ngc = paths["fordata_ran"].parent / (
        paths["fordata_ran"].stem + "_NGC.fits"
    )
    path_sgc = paths["fordata_ran"].parent / (
        paths["fordata_ran"].stem + "_SGC.fits"
    )

    split_and_write(
        ran, 'CAP', 'NGC', 'SGC',
        path_ngc, path_sgc,
        extra_meta={"STAGE": "split_ngcsgc"},
        log=log,
    )
    del ran


# =============================================================================
# Data catalogue split
# =============================================================================

def split_data_catalogue(log):
    """Split the BGS data catalogue into NGC and SGC subsets."""
    from config import BGS_CAT
    import warnings
    from astropy.io.fits.verify import VerifyWarning
    warnings.filterwarnings("ignore", category=VerifyWarning)

    log.info("=== Splitting BGS data catalogue into NGC / SGC ===")

    path_in = BGS_CAT
    if not path_in.exists():
        log.error(f"BGS catalogue not found: {path_in}")
        import sys; sys.exit(1)

    log.info(f"Reading {path_in}")
    dat = Table.read(str(path_in))
    log.info(f"Read {len(dat):,} rows")

    dat = add_cap_column(dat)

    path_ngc = path_in.parent / (path_in.stem + "_NGC.fits")
    path_sgc = path_in.parent / (path_in.stem + "_SGC.fits")

    split_and_write(
        dat, 'CAP', 'NGC', 'SGC',
        path_ngc, path_sgc,
        extra_meta={"STAGE": "split_ngcsgc"},
        log=log,
    )
    del dat


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--obs_seed", type=int, default=DEFAULT_OBS_SEED,
        help=f"Observer position seed (default: {DEFAULT_OBS_SEED})",
    )
    p.add_argument(
        "--sig_lgv", type=float, default=DEFAULT_SIG_LGV,
        help=f"log10(Vpeak) scatter in dex (default: {DEFAULT_SIG_LGV})",
    )
    p.add_argument(
        "--mocks", action="store_true",
        help="Split mock galaxy catalogues into NGC/SGC",
    )
    p.add_argument(
        "--formock", action="store_true",
        help="Split formock random catalogues into NGC/SGC",
    )
    p.add_argument(
        "--fordata", action="store_true",
        help="Split fordata random catalogue into NGC/SGC",
    )
    p.add_argument(
        "--data", action="store_true",
        help="Split the BGS data catalogue into NGC/SGC subsets",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # If no task flags given, run all four.
    run_all = not (args.mocks or args.formock or args.fordata or args.data)

    log = setup_logging("split_ngcsgc")

    # Build paths for both regions and both caps.
    paths = {}
    for reg in ("N", "S"):
        paths.update(make_paths(
            reg=reg,
            obs_seed=args.obs_seed,
            sig_lgv=args.sig_lgv,
        ))
    # Add the combined paths (no reg argument).
    paths.update(make_paths(
        obs_seed=args.obs_seed,
        sig_lgv=args.sig_lgv,
    ))

    if args.mocks or run_all:
        split_mocks(paths, log)

    if args.formock or run_all:
        split_formock_randoms(paths, log)

    if args.fordata or run_all:
        split_fordata_randoms(paths, log)

    if args.data or run_all:
        split_data_catalogue(log)

    log.info("split_ngcsgc.py complete.")


if __name__ == "__main__":
    main()
