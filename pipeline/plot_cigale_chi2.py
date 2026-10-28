"""
plot_cigale_chi2.py
===================
Plot the CIGALE CHI2 distribution for objects that have a direct TARGETID
match in the VAC versus those that required a donor (i.e. were absent from
the VAC or failed the quality cut).

This is a diagnostic to understand why ~1% of BGS galaxies lack a valid
CIGALE stellar mass estimate.  The hypothesis is that missing objects have
poor SED fits (high CHI2) concentrated toward fainter luminosities.

Output: <model_dir>/diagnostics/stellar_masses/cigale_chi2_{reg}.png

Usage
-----
python pipeline/plot_cigale_chi2.py --reg NGC SGC
python pipeline/plot_cigale_chi2.py --reg SGC --sig_lgv_ref 0.18 --sig_lgv_slope 0.30
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, join
from astropy.io import fits

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

import config
from pipeline_utils import make_paths, setup_logging

CIG_CAT = Path(
    "/global/cfs/cdirs/desi/science/gqp/y3gqp/v2.0/st_mass_MS_cigale/"
    "BGS_BRIGHT-loa-LSSv2.1-cigale-v1.0.fits"
)
CIG_LOGM_MIN = 4.0
CIG_CHI2_MAX = 17.0


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def process_region(reg, paths, args):
    mock_in = Path(paths[f"mock_{reg}"])
    outdir  = ensure_dir(Path(paths["model_dir"]) / "diagnostics" / "stellar_masses")

    # ------------------------------------------------------------------
    # Load mock (base, not augmented — just need TARGETID and Mr)
    # ------------------------------------------------------------------
    logging.info("Reading mock %s …", reg)
    mock = Table.read(str(mock_in), format="fits")
    logging.info("  %d rows", len(mock))
    mock = mock["TARGETID", "ABSMAG_RP1", "Zobs"]

    # ------------------------------------------------------------------
    # Load full CIGALE VAC (before quality cut) — TARGETID and CHI2 only
    # ------------------------------------------------------------------
    logging.info("Reading CIGALE VAC (full, no quality cut) …")
    with fits.open(str(args.cig_cat), memmap=True) as hdul:
        data = hdul[1].data
        cig_full = Table({
            "TARGETID": data["TARGETID"],
            "LOGM":     data["LOGM"],
            "CHI2":     data["CHI2"],
        })
    logging.info("  %d rows in full VAC", len(cig_full))

    # ------------------------------------------------------------------
    # Left-join: every mock row gets its CIGALE entry if it exists
    # ------------------------------------------------------------------
    logging.info("Joining …")
    joined = join(mock, cig_full, keys="TARGETID", join_type="left")
    assert len(joined) == len(mock)

    # Classify rows
    no_match  = np.asarray(joined["CHI2"].mask if hasattr(joined["CHI2"], "mask")
                           else ~np.isfinite(joined["CHI2"].filled(np.nan)
                                             if hasattr(joined["CHI2"], "filled")
                                             else joined["CHI2"]),
                           dtype=bool)
    # Use the mask from the join
    no_match = np.asarray(np.ma.getmaskarray(joined["CHI2"]))

    chi2_all   = np.asarray(joined["CHI2"].filled(np.nan)
                            if hasattr(joined["CHI2"], "filled")
                            else joined["CHI2"], dtype=float)

    has_match  = ~no_match
    passed_cut = has_match & (chi2_all < CIG_CHI2_MAX) & \
                 (np.asarray(joined["LOGM"].filled(np.nan)
                             if hasattr(joined["LOGM"], "filled")
                             else joined["LOGM"], dtype=float) > CIG_LOGM_MIN)
    failed_cut = has_match & ~passed_cut

    logging.info("  No VAC match:  %d", no_match.sum())
    logging.info("  Matched, passed cut: %d", passed_cut.sum())
    logging.info("  Matched, failed cut: %d", failed_cut.sum())

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: CHI2 distributions
    ax = axes[0]
    chi2_bins = np.linspace(0, 50, 101)
    chi2_passed = chi2_all[passed_cut]
    chi2_failed = chi2_all[failed_cut]

    ax.hist(chi2_passed, bins=chi2_bins, density=True, histtype="step",
            lw=1.5, color="tab:blue",
            label=f"Passed cut ({passed_cut.sum():,})")
    ax.hist(chi2_failed, bins=chi2_bins, density=True, histtype="step",
            lw=1.5, color="tab:red",
            label=f"Failed cut ({failed_cut.sum():,})")
    ax.axvline(CIG_CHI2_MAX, color="k", ls="--", lw=1.0,
               label=f"CHI2 cut = {CIG_CHI2_MAX}")
    ax.set_xlabel(r"CIGALE $\chi^2$")
    ax.set_ylabel("Normalised density")
    ax.set_title(f"{reg}  —  CHI2 distributions")
    ax.legend(fontsize=8)

    # Right panel: Mr distribution of no-match, failed-cut and passed-cut
    ax = axes[1]
    mr_all    = np.asarray(joined["ABSMAG_RP1"], dtype=float)
    mr_bins   = np.linspace(-24, -14, 51)

    ax.hist(mr_all[passed_cut], bins=mr_bins, density=True,
            histtype="step", lw=1.5, color="tab:blue",
            label=f"Passed cut ({passed_cut.sum():,})")
    ax.hist(mr_all[failed_cut], bins=mr_bins, density=True,
            histtype="step", lw=1.5, color="tab:red",
            label=f"Failed cut ({failed_cut.sum():,})")
    ax.hist(mr_all[no_match], bins=mr_bins, density=True,
            histtype="step", lw=1.5, color="tab:orange",
            label=f"No VAC match ({no_match.sum():,})")
    ax.set_xlabel(r"$M_r - 5\log h$")
    ax.set_ylabel("Normalised density")
    ax.set_title(f"{reg}  —  Mr distributions by CIGALE status")
    ax.legend(fontsize=8)

    fig.suptitle(f"{reg}  —  CIGALE CHI2 and completeness diagnostic",
                 fontsize=11)
    fig.tight_layout()
    outfile = outdir / f"cigale_chi2_{reg}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", outfile)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot CIGALE CHI2 distributions for matched vs unmatched objects."
    )
    p.add_argument("--reg", nargs="+", default=["NGC", "SGC"],
                   choices=["NGC", "SGC"])
    p.add_argument("--obs_seed",      type=int,   default=0)
    p.add_argument("--sig_lgv_ref",   type=float, default=config.DEFAULT_SIG_LGV_REF)
    p.add_argument("--sig_lgv_slope", type=float, default=config.DEFAULT_SIG_LGV_SLOPE)
    p.add_argument("--f_colour",      type=float, default=config.DEFAULT_F_COLOUR)
    p.add_argument("--alpha",         type=float, default=config.DEFAULT_ALPHA)
    p.add_argument("--cig_cat",  type=Path, default=CIG_CAT,
                   help="Path to CIGALE VAC FITS file")
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging("plot_cigale_chi2")
    root = logging.getLogger()
    root.setLevel(log.level)
    for h in log.handlers:
        root.addHandler(h)
    logging.info("Regions: %s", args.reg)

    paths = make_paths(
        obs_seed=args.obs_seed,
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=args.alpha,
    )

    for reg in args.reg:
        logging.info("=" * 60)
        logging.info("Region: %s", reg)
        process_region(reg, paths, args)

    logging.info("Done.")


if __name__ == "__main__":
    main()
