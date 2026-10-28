"""
plot_mstar_ssfr.py
==================
Diagnostic plot of stellar mass versus specific star-formation rate (sSFR)
for the BGS sample, using CIGALE and FastSpecFit (FSF) derived quantities
from the augmented mock catalogues.

Produces one figure per region (NGC, SGC) with two panels:
  Left:  CIGALE  log M* vs log(SFR) - log M*  (sSFR proxy)
  Right: FSF     log M* vs log(SFR/M*)

The plots are analogous to Elena's Mr vs rest-frame colour diagnostic and
are used to guide the choice of sSFR cut that separates the star-forming
main sequence from the quiescent population.

A candidate diagonal cut of the form
    log(sSFR) = ssfr_cut + ssfr_slope * (log M* - mstar_pivot)
is overlaid on each panel.  The default values are placeholders — adjust
on the CLI after inspecting the plots.

Output saved to <model_dir>/diagnostics/stellar_masses/mstar_ssfr_{reg}.png

Usage
-----
python pipeline/plot_mstar_ssfr.py --reg NGC SGC
python pipeline/plot_mstar_ssfr.py --reg NGC --sig_lgv_ref 0.18 --sig_lgv_slope 0.30 \\
    --ssfr_cut -11.0 --ssfr_slope -0.3 --mstar_pivot 10.5
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

import config
from pipeline_utils import make_paths, setup_logging

# Default candidate cut parameters (adjust after inspecting the plots)
DEFAULT_SSFR_CUT    = -11.0   # log(sSFR) at mstar_pivot
DEFAULT_SSFR_SLOPE  =  0.0    # tilt of the cut in log M* space
DEFAULT_MSTAR_PIVOT = 10.5    # log M* at which the cut is anchored


def load_augmented(paths, reg):
    mock_in = Path(paths[f"mock_{reg}"])
    aug = mock_in.with_name(mock_in.stem + "_augmented.fits")
    if not aug.exists():
        raise FileNotFoundError(
            f"Augmented mock not found: {aug}\nRun add_stellar_masses.py first."
        )
    logging.info("Reading %s …", aug)
    t = Table.read(str(aug))
    logging.info("  %d rows", len(t))
    return t


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def candidate_cut(logm, ssfr_cut, ssfr_slope, mstar_pivot):
    """Return log(sSFR) threshold at each log M* value."""
    return ssfr_cut + ssfr_slope * (logm - mstar_pivot)


def plot_panel(ax, logm, logssfr, title, ssfr_cut, ssfr_slope, mstar_pivot,
               hexbin_cmap="Blues"):
    """Draw one M* vs sSFR hexbin panel with candidate cut overlaid."""
    finite = np.isfinite(logm) & np.isfinite(logssfr)
    logm_f    = logm[finite]
    logssfr_f = logssfr[finite]

    hb = ax.hexbin(logm_f, logssfr_f,
                   gridsize=150, bins="log", mincnt=1,
                   cmap=hexbin_cmap, zorder=1)
    plt.colorbar(hb, ax=ax, label="N")

    # Candidate cut
    mstar_range = np.linspace(7, 13, 200)
    cut_line    = candidate_cut(mstar_range, ssfr_cut, ssfr_slope, mstar_pivot)
    ax.plot(mstar_range, cut_line, "r-", lw=1.5, zorder=2,
            label=f"cut: {ssfr_cut:+.1f}{ssfr_slope:+.2f}"
                  f"(logM*−{mstar_pivot:.1f})")

    # Star-forming main sequence reference (rough empirical line)
    ms_ssfr = -0.35 * (mstar_range - 10.5) - 9.8
    ax.plot(mstar_range, ms_ssfr, "k--", lw=1.0, alpha=0.5,
            zorder=2, label="approx. main sequence")

    ax.set_xlim(7, 13)
    ax.set_ylim(-14, -7)
    ax.set_xlabel(r"$\log M_\star / M_\odot$")
    ax.set_ylabel(r"$\log\,\mathrm{sSFR}\ [\mathrm{yr}^{-1}]$")
    ax.set_title(title)
    ax.legend(fontsize=8)

    # Report fractions above/below cut
    cut_vals = candidate_cut(logm_f, ssfr_cut, ssfr_slope, mstar_pivot)
    sf_frac  = (logssfr_f > cut_vals).sum() / len(logm_f)
    ax.text(0.03, 0.05,
            f"SF: {100*sf_frac:.1f}%  Q: {100*(1-sf_frac):.1f}%",
            transform=ax.transAxes, fontsize=8, va="bottom")


def process_region(reg, paths, args):
    mock   = load_augmented(paths, reg)
    outdir = ensure_dir(Path(paths["model_dir"]) / "diagnostics" / "stellar_masses")

    # CIGALE quantities
    logm_cig    = np.asarray(mock["LOGM"],   dtype=float)
    logsfr_cig  = np.asarray(mock["LOGSFR"], dtype=float)
    logssfr_cig = logsfr_cig - logm_cig

    # FSF quantities: SFR is linear, not log; convert carefully
    logmstar_fsf = np.asarray(mock["LOGMSTAR"], dtype=float)
    sfr_fsf      = np.asarray(mock["SFR"],      dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logssfr_fsf = np.where(
            sfr_fsf > 0,
            np.log10(sfr_fsf) - logmstar_fsf,
            np.nan
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_panel(
        axes[0], logm_cig, logssfr_cig,
        title=f"{reg}  —  CIGALE",
        ssfr_cut=args.ssfr_cut,
        ssfr_slope=args.ssfr_slope,
        mstar_pivot=args.mstar_pivot,
        hexbin_cmap="Blues",
    )
    plot_panel(
        axes[1], logmstar_fsf, logssfr_fsf,
        title=f"{reg}  —  FSF",
        ssfr_cut=args.ssfr_cut,
        ssfr_slope=args.ssfr_slope,
        mstar_pivot=args.mstar_pivot,
        hexbin_cmap="Greens",
    )

    fig.suptitle(f"{reg}  —  stellar mass vs sSFR", fontsize=11)
    fig.tight_layout()

    outfile = outdir / f"mstar_ssfr_{reg}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", outfile)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot M* vs sSFR diagnostic for augmented mock catalogues."
    )
    p.add_argument("--reg", nargs="+", default=["NGC", "SGC"],
                   choices=["NGC", "SGC"])
    p.add_argument("--obs_seed",      type=int,   default=0)
    p.add_argument("--sig_lgv_ref",   type=float, default=config.DEFAULT_SIG_LGV_REF)
    p.add_argument("--sig_lgv_slope", type=float, default=config.DEFAULT_SIG_LGV_SLOPE)
    p.add_argument("--f_colour",      type=float, default=config.DEFAULT_F_COLOUR)
    p.add_argument("--alpha",         type=float, default=config.DEFAULT_ALPHA)
    p.add_argument("--ssfr_cut",    type=float, default=DEFAULT_SSFR_CUT,
                   help="log(sSFR) threshold at mstar_pivot (default: %(default)s)")
    p.add_argument("--ssfr_slope",  type=float, default=DEFAULT_SSFR_SLOPE,
                   help="Tilt of sSFR cut per dex in log M* (default: %(default)s)")
    p.add_argument("--mstar_pivot", type=float, default=DEFAULT_MSTAR_PIVOT,
                   help="log M* at which sSFR cut is anchored (default: %(default)s)")
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging("plot_mstar_ssfr")
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
