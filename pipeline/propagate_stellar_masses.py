"""
propagate_stellar_masses.py
===========================
Propagate stellar mass and SFR columns from the augmented master BGS
catalogue to a list of other catalogues (mocks, randoms, data subsets)
via a TARGETID left-join.

Every galaxy in the target catalogues should have a TARGETID present in
the master catalogue (randoms are clones of data galaxies; mocks inherit
their TARGETID from the matched random).  Any rows without a match are
left with masked/NaN values and flagged in the output log.

The script reads one target file at a time, joins, and writes the result
back to the same path (atomically via a temp file), freeing memory before
moving to the next file.  This means the clustering pipeline can continue
to read the same file paths without modification.

Columns copied from master
--------------------------
LOGM, LOGSFR, cigflag, LOGMSTAR, SFR, fsfflag

Usage
-----
# Augment a list of files in-place:
python pipeline/propagate_stellar_masses.py \\
    --master ~/DirectSHAM/data/6thOct/BGS_BRIGHT_6thOct_augmented.fits \\
    --files  ~/DirectSHAM/data/6thOct/BGS_BRIGHT_6thOct_NGC.fits \\
             ~/DirectSHAM/data/6thOct/BGS_BRIGHT_6thOct_SGC.fits \\
             ~/DirectSHAM/data/6thOct/rancat_fordata_NGC.fits \\
             ~/DirectSHAM/data/6thOct/rancat_fordata_SGC.fits \\
             ~/DirectSHAM/data/6thOct/N_neigh_lum50/seed000_sig_lgv0.18_0.30_fc0.00_al0.00/mock_NGC.fits \\
             ~/DirectSHAM/data/6thOct/N_neigh_lum50/seed000_sig_lgv0.18_0.30_fc0.00_al0.00/mock_SGC.fits

# Dry run (check match rates without writing):
python pipeline/propagate_stellar_masses.py --master ... --files ... --dry_run
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.table import Table, join

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
from pipeline_utils import setup_logging

# Columns to carry from the master catalogue (plus TARGETID as join key)
COPY_COLS = ["TARGETID", "LOGM", "LOGSFR", "cigflag",
             "LOGMSTAR", "SFR", "fsfflag"]


def load_master(path):
    """Read only the needed columns from the master augmented catalogue."""
    logging.info("Reading master catalogue: %s", path)
    master = Table.read(str(path), format="fits")
    # Keep only the columns we need — saves memory
    missing = [c for c in COPY_COLS if c not in master.colnames]
    if missing:
        raise ValueError(f"Master catalogue missing columns: {missing}")
    master = master[COPY_COLS]
    logging.info("  %d rows, %d columns retained", len(master), len(COPY_COLS))
    return master


def process_file(path, master, dry_run=False):
    """
    Left-join stellar mass columns from master onto the target file and
    write back in-place (via a temp file for safety).
    """
    path = Path(path)
    if not path.exists():
        logging.error("File not found: %s", path)
        return False

    logging.info("-" * 60)
    logging.info("Processing: %s", path.name)

    cat = Table.read(str(path), format="fits")
    n_rows = len(cat)
    logging.info("  %d rows", n_rows)

    # Check TARGETID is present
    if "TARGETID" not in cat.colnames:
        logging.error("  No TARGETID column — skipping")
        return False

    # Drop any existing stellar mass columns to avoid conflicts
    for col in COPY_COLS[1:]:   # skip TARGETID
        if col in cat.colnames:
            cat.remove_column(col)
            logging.info("  Removed existing column: %s", col)

    # Left-join
    cat = join(cat, master, keys="TARGETID", join_type="left")
    if len(cat) != n_rows:
        logging.error("  Row count changed after join: %d -> %d", n_rows, len(cat))
        return False

    # Report match rate
    n_unmatched = int(np.ma.getmaskarray(cat["LOGM"]).sum())
    logging.info("  Matched: %d / %d  (unmatched: %d, %.2f%%)",
                 n_rows - n_unmatched, n_rows,
                 n_unmatched, 100.0 * n_unmatched / n_rows)

    if dry_run:
        logging.info("  DRY RUN — not writing")
        return True

    # Write atomically: temp file in same directory, then rename
    tmp = path.with_suffix(".fits.tmp")
    try:
        cat = Table(cat)   # strip masked-table wrappers
        cat.write(str(tmp), overwrite=True, format="fits")
        os.replace(str(tmp), str(path))
        logging.info("  Written: %s", path.name)
    except Exception as e:
        logging.error("  Write failed: %s", e)
        if tmp.exists():
            tmp.unlink()
        return False

    return True


def parse_args():
    p = argparse.ArgumentParser(
        description="Propagate stellar masses from master BGS catalogue to "
                    "target files via TARGETID join.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--master", type=Path, required=True,
                   help="Augmented master BGS catalogue with LOGM, LOGMSTAR etc.")
    p.add_argument("--files", type=Path, nargs="+", required=True,
                   help="Target files to augment in-place")
    p.add_argument("--dry_run", action="store_true",
                   help="Report match rates without writing any files")
    p.add_argument("--cols", nargs="+", default=None,
                   help="Override default column list to copy "
                        "(default: LOGM LOGSFR cigflag LOGMSTAR SFR fsfflag)")
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging("propagate_stellar_masses")
    root = logging.getLogger()
    root.setLevel(log.level)
    for h in log.handlers:
        root.addHandler(h)

    if args.cols:
        global COPY_COLS
        COPY_COLS = ["TARGETID"] + args.cols

    if not args.master.exists():
        logging.error("Master catalogue not found: %s", args.master)
        sys.exit(1)

    master = load_master(args.master)

    n_ok = 0
    n_fail = 0
    for f in args.files:
        ok = process_file(f, master, dry_run=args.dry_run)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
        # Explicitly keep master in memory but free the per-file table
        # (process_file returns after write so cat goes out of scope)

    logging.info("=" * 60)
    logging.info("Done: %d succeeded, %d failed", n_ok, n_fail)
    if args.dry_run:
        logging.info("(dry run — no files modified)")


if __name__ == "__main__":
    main()
