"""
confusion_matrix_sfq.py
=======================
For each CIGALE stellar mass threshold sample, compute the confusion matrix
between the sSFR-based SF/Q classification (CIGALE log sSFR cut at -11)
and the colour-based red/blue classification (tilted g-r cut).

Reads from the clustering catalogue files (already cut to the right
stellar mass and redshift limits) which carry LOGM, LOGSFR, REST_GMR_0P1
and ABSMAG_RP1.

Usage
-----
python pipeline/confusion_matrix_sfq.py --sig_lgv_ref 0.18 --sig_lgv_slope 0.30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import make_paths, setup_logging
from config import DATA_DATE_DIR, DEFAULT_SIG_LGV_REF, DEFAULT_SIG_LGV_SLOPE

# Adopted stellar mass thresholds and redshift limits
SAMPLES = [
    dict(cig_cut=11.10, zmin=0.02, zmax=0.42),
    dict(cig_cut=10.60, zmin=0.02, zmax=0.34),
    dict(cig_cut=10.10, zmin=0.02, zmax=0.27),
    dict(cig_cut= 9.50, zmin=0.02, zmax=0.21),
    dict(cig_cut= 8.90, zmin=0.02, zmax=0.17),
]

LOGSSFR_CUT = -11.0   # CIGALE passive/SF boundary


def colour_is_red(gmr, mr):
    """Tilted red/blue cut: red if g-r >= 0.21 - 0.03 * Mr."""
    return gmr >= 0.21 - 0.03 * mr


def confusion_matrix(sample, paths, log):
    cig_cut = sample["cig_cut"]
    zmin    = sample["zmin"]
    zmax    = sample["zmax"]

    # Read NGC + SGC clustering catalogues (already stellar-mass and z cut)
    tag     = f"Mcig{cig_cut}_z{zmin:.2f}-{zmax:.2f}"
    cat_dir = Path(paths["model_dir"]) / "clustering" / tag / "catalogues"

    tables = []
    for cap in ("NGC", "SGC"):
        p = cat_dir / f"BGS_BRIGHT_{cap}_clustering.dat.fits"
        if not p.exists():
            log.warning("Not found: %s", p)
            continue
        tables.append(Table.read(str(p)))

    if not tables:
        log.warning("No catalogues found for %s — skipping", tag)
        return

    cat = vstack(tables)
    n   = len(cat)
    log.info("%s: %d objects", tag, n)

    # Check required columns
    for col in ("LOGM", "LOGSFR", "REST_GMR_0P1", "ABSMAG_RP1"):
        if col not in cat.colnames:
            log.error("Missing column %s in %s", col, tag)
            return

    logm   = np.asarray(cat["LOGM"],        dtype=float)
    logsfr = np.asarray(cat["LOGSFR"],       dtype=float)
    gmr    = np.asarray(cat["REST_GMR_0P1"], dtype=float)
    mr     = np.asarray(cat["ABSMAG_RP1"],   dtype=float)

    # sSFR classification
    logssfr  = logsfr - logm
    is_sf    = logssfr >= LOGSSFR_CUT
    is_pass  = ~is_sf
    n_sf     = is_sf.sum()
    n_pass   = is_pass.sum()

    # Colour classification
    is_blue  = ~colour_is_red(gmr, mr)
    is_red   = colour_is_red(gmr, mr)
    n_blue   = is_blue.sum()
    n_red    = is_red.sum()

    # Confusion matrix
    sf_blue   = (is_sf   & is_blue).sum()
    sf_red    = (is_sf   & is_red).sum()
    pass_blue = (is_pass & is_blue).sum()
    pass_red  = (is_pass & is_red).sum()

    # Print results
    print(f"\n{'='*60}")
    print(f"Sample: log M* > {cig_cut:.2f},  "
          f"{zmin:.2f} <= z <= {zmax:.2f},  N={n:,}")
    print(f"  sSFR cut: log(sSFR) = {LOGSSFR_CUT}")
    print(f"  SF: {n_sf:,} ({100*n_sf/n:.1f}%)   "
          f"Passive: {n_pass:,} ({100*n_pass/n:.1f}%)")
    print(f"  Blue: {n_blue:,} ({100*n_blue/n:.1f}%)   "
          f"Red: {n_red:,} ({100*n_red/n:.1f}%)")
    print()
    print(f"  Confusion matrix (rows=sSFR class, cols=colour class):")
    print(f"  {'':12s}  {'Blue':>10s}  {'Red':>10s}  {'Total':>10s}")
    print(f"  {'SF':12s}  {sf_blue:>10,}  {sf_red:>10,}  {n_sf:>10,}")
    print(f"  {'Passive':12s}  {pass_blue:>10,}  {pass_red:>10,}  {n_pass:>10,}")
    print(f"  {'Total':12s}  {n_blue:>10,}  {n_red:>10,}  {n:>10,}")
    print()
    print(f"  Purity   (SF=blue): {100*sf_blue/n_blue:.1f}%  "
          f"(Passive=red): {100*pass_red/n_red:.1f}%")
    print(f"  Recovery (SF=blue): {100*sf_blue/n_sf:.1f}%  "
          f"(Passive=red): {100*pass_red/n_pass:.1f}%")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sig_lgv_ref",   type=float, default=DEFAULT_SIG_LGV_REF)
    p.add_argument("--sig_lgv_slope", type=float, default=DEFAULT_SIG_LGV_SLOPE)
    p.add_argument("--obs_seed",      type=int,   default=0)
    p.add_argument("--f_colour",      type=float, default=0.0)
    p.add_argument("--alpha",         type=float, default=0.0)
    p.add_argument("--logssfr_cut",   type=float, default=LOGSSFR_CUT,
                   help=f"log10 sSFR cut for SF/passive split "
                        f"(default: {LOGSSFR_CUT})")
    return p.parse_args()


def main():
    args = parse_args()
    log  = setup_logging("confusion_matrix_sfq")

    global LOGSSFR_CUT
    LOGSSFR_CUT = args.logssfr_cut

    paths = make_paths(obs_seed=args.obs_seed,
                       sig_lgv_ref=args.sig_lgv_ref,
                       sig_lgv_slope=args.sig_lgv_slope,
                       f_colour=args.f_colour,
                       alpha=args.alpha)

    for sample in SAMPLES:
        confusion_matrix(sample, paths, log)

    print("\nDone.")


if __name__ == "__main__":
    main()
