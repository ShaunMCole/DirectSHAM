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

from config import DEFAULT_OBS_SEED, DEFAULT_SIG_LGV_REF, DEFAULT_SIG_LGV_SLOPE, DEFAULT_F_COLOUR, DEFAULT_ALPHA
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

    FSKY in the metadata is scaled by the fraction of rows in each cap so
    that the per-cap sky area is correct.  This requires NMULT to be the
    same for both caps (enforced upstream in split_formock_randoms).

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
    ngc = table[table[col] == val_ngc].copy()
    sgc = table[table[col] == val_sgc].copy()

    # Update FSKY to reflect each cap's fraction of the total footprint.
    # Since randoms are uniform and NMULT is identical for N and S, the
    # row count ratio equals the sky area ratio.
    n_total = len(ngc) + len(sgc)
    if 'FSKY' in table.meta and n_total > 0:
        fsky_total = table.meta['FSKY']
        fsky_ngc   = fsky_total * len(ngc) / n_total
        fsky_sgc   = fsky_total * len(sgc) / n_total
        ngc.meta['FSKY'] = fsky_ngc
        sgc.meta['FSKY'] = fsky_sgc
        msg = (f"FSKY: combined={fsky_total:.6f}  "
               f"NGC={fsky_ngc:.6f}  SGC={fsky_sgc:.6f}")
        if log:
            log.info(msg)
        else:
            print(msg)

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

    # Sum N and S FSKY explicitly before vstack — metadata_conflicts='silent'
    # would otherwise silently drop one of the two values.
    fsky_N = ranN.meta.get('FSKY', 0.0)
    fsky_S = ranS.meta.get('FSKY', 0.0)
    fsky_combined = fsky_N + fsky_S
    log.info(f"Formock fsky: N={fsky_N:.6f}  S={fsky_S:.6f}  combined={fsky_combined:.6f}")

    log.info("Concatenating N + S")
    ran = vstack([ranN, ranS], metadata_conflicts='silent')
    ran.meta['FSKY'] = fsky_combined
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
    """
    Split the fordata random catalogues (N and S) into NGC and SGC subsets.

    The N and S fordata files do not carry FSKY in their headers.  The correct
    per-region fsky values come from ca.selection(reg)['area'], which returns
    the actual observed footprint sky fraction.  These are combined and then
    apportioned to NGC/SGC by row count (randoms are uniform, so row fraction
    equals area fraction).
    """
    import catalogue_analysis as ca
    log.info("=== Splitting fordata randoms into NGC / SGC ===")

    path_N = paths["fordata_ran_N"]
    # fordata_ran_S is only present when make_paths is called with reg='S';
    # derive it from the N path since the naming convention is consistent.
    path_S = path_N.parent / path_N.name.replace('_N.fits', '_S.fits')

    for p in (path_N, path_S):
        if not p.exists():
            log.error(f"Input file not found: {p}")
            sys.exit(1)

    log.info(f"Reading {path_N}")
    ranN = Table.read(path_N)
    log.info(f"Reading {path_S}")
    ranS = Table.read(path_S)

    # Compute combined fsky from ca.selection() — these are the correct
    # fordata footprint sky fractions (smaller than formock, which uses a
    # more liberal mask).
    fsky_N = ca.selection('N')['area']
    fsky_S = ca.selection('S')['area']
    fsky_combined = fsky_N + fsky_S
    log.info(f"Fordata fsky: N={fsky_N:.6f}  S={fsky_S:.6f}  combined={fsky_combined:.6f}")

    # Store combined FSKY in metadata so split_and_write can apportion it.
    log.info("Concatenating N + S")
    ran = vstack([ranN, ranS], metadata_conflicts='silent')
    ran.meta['FSKY'] = fsky_combined
    del ranN, ranS

    log.info("Assigning NGC/SGC by galactic latitude")
    ran = add_cap_column(ran)

    path_ngc = path_N.parent / path_N.name.replace('_N.fits', '_NGC.fits')
    path_sgc = path_N.parent / path_N.name.replace('_N.fits', '_SGC.fits')

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
        "--sig_lgv_ref", type=float, default=DEFAULT_SIG_LGV_REF,
        help=f"Vpeak scatter at pivot Vpeak (default: {DEFAULT_SIG_LGV_REF})",
    )
    p.add_argument(
        "--sig_lgv_slope", type=float, default=DEFAULT_SIG_LGV_SLOPE,
        help=f"Gradient of scatter with log10(Vpeak) (default: {DEFAULT_SIG_LGV_SLOPE})",
    )
    p.add_argument(
        "--f_colour", type=float, default=DEFAULT_F_COLOUR,
        help=f"f_colour used when making the checkpoint (default: {DEFAULT_F_COLOUR})",
    )
    p.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"alpha used when making the mock (default: {DEFAULT_ALPHA})",
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
            sig_lgv_ref=args.sig_lgv_ref,
            sig_lgv_slope=args.sig_lgv_slope,
            f_colour=args.f_colour,
            alpha=args.alpha,
        ))
    # Add the combined paths (no reg argument).
    paths.update(make_paths(
        obs_seed=args.obs_seed,
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=args.alpha,
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
