"""
plot_nz_mstar.py
================
Plot comoving number density n(z) [h^3 Mpc^-3] for the BGS data sample
using both absolute magnitude thresholds (reference) and CIGALE/FSF stellar
mass thresholds, to guide the choice of stellar mass cuts and upper redshift
limits for clustering measurements.

Reads the augmented catalogues (NGC + SGC combined) and produces a single
figure with three panels, analogous to the nz_by_magcut plot in inspect_mocks.py.

Output: <model_dir>/diagnostics/stellar_masses/nz_mstar.png

Usage
-----
python pipeline/plot_nz_mstar.py --sig_lgv_ref 0.18 --sig_lgv_slope 0.30
python pipeline/plot_nz_mstar.py --mr_cuts -21.5 -20.5 -19.5 -18.5 -17.5 \\
    --mstar_cuts 11.0 10.5 10.0 9.5 9.0
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

import config
from pipeline_utils import make_paths, setup_logging

# Analysis cosmology (matches rest of pipeline)
cosmo = FlatLambdaCDM(H0=100, Om0=0.313)

FSKY_TOT = None   # computed from fordata random headers at runtime

DEFAULT_MR_CUTS       = [-21.5, -20.5, -19.5, -18.5, -17.5]
DEFAULT_MSTAR_CUTS    = [11.0, 10.5, 10.0, 9.5, 9.0]
DEFAULT_MSTAR_CUTS_FSF = None   # defaults to same as --mstar_cuts if not set

Z_MIN  = 0.02
Z_MAX  = 0.50
N_BINS = 48
EPS    = 1e-25


def load_data(paths, reg, cat=None):
    if cat is not None:
        # Use master augmented data file directly
        path = Path(cat)
        if not path.exists():
            raise FileNotFoundError(f"Catalogue not found: {path}")
    else:
        mock_in = Path(paths[f"mock_{reg}"])
        path = mock_in.with_name(mock_in.stem + "_augmented.fits")
        if not path.exists():
            raise FileNotFoundError(f"Augmented mock not found: {path}")
    logging.info("Reading %s …", path)
    t = Table.read(str(path), format="fits")
    cols = [c for c in ["Z", "Zobs", "ABSMAG_RP1", "LOGM", "LOGMSTAR", "WEIGHT"]
            if c in t.colnames]
    return t[cols]


def nz_line(ax, z, vals, threshold, weights, zbins, vol, color, label,
            annotate_x=None, annotate_text=None, fontsize=6):
    """Plot n(z) as a line for objects with vals >= threshold."""
    mask = np.isfinite(vals) & (vals >= threshold)
    h, _ = np.histogram(z[mask], bins=zbins,
                        weights=weights[mask] if weights is not None else None)
    nz = np.log10(h / vol + EPS)
    zcen = 0.5 * (zbins[:-1] + zbins[1:])
    ax.plot(zcen, nz, color=color, lw=0.8, label=label)
    if annotate_x is not None and annotate_text is not None:
        # Find nz value at annotation redshift
        idx = np.argmin(np.abs(zcen - annotate_x))
        if np.isfinite(nz[idx]):
            ax.annotate(annotate_text, xy=(zcen[idx], nz[idx]),
                        xytext=(3, 0), textcoords="offset points",
                        fontsize=fontsize, color=color, va="center")


def nz_scatter_mr(ax, z, mr, threshold, weights, zbins, vol, color, label,
                  annotate_x=None, annotate_text=None, fontsize=6):
    """Plot n(z) scatter points for objects with mr < threshold (Mr is negative)."""
    mask = np.isfinite(mr) & (mr < threshold)
    h, _ = np.histogram(z[mask], bins=zbins,
                        weights=weights[mask] if weights is not None else None)
    nz = np.log10(h / vol + EPS)
    zcen = 0.5 * (zbins[:-1] + zbins[1:])
    ax.scatter(zcen, nz, s=2, color=color, label=label)
    if annotate_x is not None and annotate_text is not None:
        idx = np.argmin(np.abs(zcen - annotate_x))
        if np.isfinite(nz[idx]):
            ax.annotate(annotate_text, xy=(zcen[idx], nz[idx]),
                        xytext=(3, 0), textcoords="offset points",
                        fontsize=fontsize, color=color, va="center")


def make_plot(z, mr, logm_cig, logm_fsf, weights, zbins, vol,
              mr_cuts, mstar_cuts, mstar_cuts_fsf, outfile):

    mr_colors  = plt.cm.Blues_r( np.linspace(0.2, 0.7, len(mr_cuts)))
    cig_colors = plt.cm.Reds_r(  np.linspace(0.2, 0.7, len(mstar_cuts)))
    fsf_colors = plt.cm.Greens_r(np.linspace(0.2, 0.7, len(mstar_cuts_fsf)))

    xlim = (Z_MIN, 0.50)
    ylim = (-4.0, -1.25)

    FSIZE   = 7
    FSIZE_S = 6
    plt.rcParams.update({"font.size": FSIZE, "axes.labelsize": FSIZE,
                         "xtick.labelsize": FSIZE_S, "ytick.labelsize": FSIZE_S})

    fig, ax = plt.subplots(figsize=(3.31, 2.8))

    # Mr thresholds — blue scatter (no annotation)
    for cut, col in zip(mr_cuts, mr_colors):
        nz_scatter_mr(ax, z, mr, cut, weights, zbins, vol, col, label=None)

    # CIGALE M* thresholds — red lines (no annotation)
    for cut, col in zip(mstar_cuts, cig_colors):
        nz_line(ax, z, logm_cig, cut, weights, zbins, vol, col, label=None)

    # FSF M* thresholds — green lines (no annotation)
    for cut, col in zip(mstar_cuts_fsf, fsf_colors):
        nz_line(ax, z, logm_fsf, cut, weights, zbins, vol, col, label=None)

    # -------------------------------------------------------------------
    # Bespoke legend: header + 5 data rows
    # Header:  ●  Mr          CIGALE   FSF
    # Row i:   ●  Mr<X.X  log M★> Y.YY, Z.ZZ
    # -------------------------------------------------------------------
    mr_labels = [rf"$M_r < {c:.1f}$" for c in mr_cuts]

    # Column x-positions in axes fraction
    x_dot  = 0.45   # dot marker
    x_mr   = 0.48   # Mr label (left-aligned)
    x_lm   = 0.768  # "log M★>" right-aligned; CIGALE left-aligned
    x_fsf  = x_lm + 0.100  # FSF number (left-aligned, no comma)

    y0  = 0.99   # top of legend
    dy  = 0.048  # row spacing
    fs  = 5.5    # slightly smaller font

    # Header row — centred over each column
    ax.text(x_lm + 0.034, y0, "CIGALE", transform=ax.transAxes,
            ha="center", va="top", fontsize=fs, color=cig_colors[0],
            fontstyle="italic")
    ax.text(x_fsf + 0.025, y0, "FSF",   transform=ax.transAxes,
            ha="center", va="top", fontsize=fs, color=fsf_colors[0],
            fontstyle="italic")

    for i, (mr_lbl, mr_col, cig_cut, cig_col, fsf_cut, fsf_col) in enumerate(
            zip(mr_labels, mr_colors, mstar_cuts, cig_colors,
                mstar_cuts_fsf, fsf_colors)):
        y = y0 - (i + 1) * dy

        # Dot — use same y as text (va="center" equivalent at midpoint)
        ax.plot([x_dot], [y - 0.015], 'o', ms=2, color=mr_col,
                transform=ax.transAxes, clip_on=False)
        # Mr label
        ax.text(x_mr,  y, mr_lbl,               transform=ax.transAxes,
                ha="left", va="top", fontsize=fs, color=mr_col)
        # log M* > (right-aligned so it butts up cleanly against CIGALE number)
        ax.text(x_lm,  y, r"$\log M_\star\!>$", transform=ax.transAxes,
                ha="right", va="top", fontsize=fs, color="k")
        # CIGALE number (left-aligned from same anchor)
        ax.text(x_lm,  y, rf"${cig_cut:.2f}$",  transform=ax.transAxes,
                ha="left", va="top", fontsize=fs, color=cig_col)
        # FSF number (no comma)
        ax.text(x_fsf, y, rf"${fsf_cut:.2f}$",  transform=ax.transAxes,
                ha="left", va="top", fontsize=fs, color=fsf_col)

    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$\log_{10}\,\bar{n}(z)\ [h^3\,\mathrm{Mpc}^{-3}]$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.tight_layout(pad=0.5)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.rcdefaults()
    plt.close(fig)
    logging.info("Saved %s", outfile)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot n(z) [h^3 Mpc^-3] for Mr and M* threshold samples."
    )
    p.add_argument("--obs_seed",      type=int,   default=0)
    p.add_argument("--sig_lgv_ref",   type=float, default=config.DEFAULT_SIG_LGV_REF)
    p.add_argument("--sig_lgv_slope", type=float, default=config.DEFAULT_SIG_LGV_SLOPE)
    p.add_argument("--f_colour",      type=float, default=config.DEFAULT_F_COLOUR)
    p.add_argument("--alpha",         type=float, default=config.DEFAULT_ALPHA)
    p.add_argument("--mr_cuts",    nargs="+", type=float, default=DEFAULT_MR_CUTS,
                   help="Absolute magnitude thresholds")
    p.add_argument("--mstar_cuts", nargs="+", type=float, default=DEFAULT_MSTAR_CUTS,
                   help="CIGALE log M* thresholds")
    p.add_argument("--mstar_cuts_fsf", nargs="+", type=float, default=None,
                   help="FSF log M* thresholds (default: same as --mstar_cuts)")
    p.add_argument("--fsky", type=float, default=None,
                   help="Sky fraction for volume calculation (default: read from fordata random headers)")
    p.add_argument("--cat", type=Path, default=None,
                   help="Read from this augmented data catalogue instead of mock "
                        "(e.g. BGS_BRIGHT_6thOct_augmented.fits). "
                        "NGC and SGC are read from a single file.")
    p.add_argument("--outfile", type=Path, default=None,
                   help="Override output file path")
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging("plot_nz_mstar")
    root = logging.getLogger()
    root.setLevel(log.level)
    for h in log.handlers:
        root.addHandler(h)

    paths = make_paths(
        obs_seed=args.obs_seed,
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=args.alpha,
    )

    # Load data
    if args.cat is not None:
        # Single master augmented data file
        data = load_data(paths, reg=None, cat=args.cat)
        logging.info("Loaded %d rows from %s", len(data), args.cat.name)
    else:
        # Load and combine NGC + SGC mock augmented files
        tables = []
        for reg in ("NGC", "SGC"):
            try:
                tables.append(load_data(paths, reg))
            except FileNotFoundError as e:
                logging.warning("%s", e)
        if not tables:
            raise RuntimeError("No augmented catalogues found.")
        data = vstack(tables)
        logging.info("Combined: %d rows", len(data))

    # Use Z for data catalogue, Zobs for mocks
    z_col = "Z" if "Z" in data.colnames and args.cat is not None else \
            "Zobs" if "Zobs" in data.colnames else "Z"
    z        = np.asarray(data[z_col],        dtype=float)
    mr       = np.asarray(data["ABSMAG_RP1"], dtype=float)
    logm_cig = np.asarray(data["LOGM"],       dtype=float)
    logm_fsf = np.asarray(data["LOGMSTAR"],   dtype=float)
    weights  = (np.asarray(data["WEIGHT"], dtype=float)
                if "WEIGHT" in data.colnames else None)

    # Redshift bins and shell volumes
    # Read FSKY from the fordata random headers — correct observed footprint area
    from astropy.io.fits import getheader
    fsky = args.fsky
    if fsky is None:
        fsky = 0.0
        if args.cat is not None:
            # Read AREA_N + AREA_S from the augmented data file header
            hdr = getheader(str(args.cat), ext=1)
            area_n = hdr.get("AREA_N", 0.0)
            area_s = hdr.get("AREA_S", 0.0)
            fsky = area_n + area_s
            logging.info("FSKY from AREA_N+AREA_S in %s: %.6f (N=%.6f S=%.6f)",
                         args.cat.name, fsky, area_n, area_s)
        else:
            for reg in ("NGC", "SGC"):
                ran_path = Path(paths[f"formock_ran_{reg}"])
                if ran_path.exists():
                    hdr = getheader(str(ran_path), ext=1)
                    fsky += hdr.get("FSKY", 0.0)
                else:
                    logging.warning("Random not found: %s", ran_path)
            logging.info("FSKY from formock headers: %.6f", fsky)

    zbins = np.linspace(Z_MIN, Z_MAX, N_BINS + 1)
    dc    = cosmo.comoving_distance(zbins).value   # Mpc/h
    vol   = (4.0 * np.pi * fsky / 3.0) * (dc[1:]**3 - dc[:-1]**3)

    if args.outfile is not None:
        outfile = args.outfile
        outfile.parent.mkdir(parents=True, exist_ok=True)
    elif args.cat is not None:
        outdir = args.cat.parent / "diagnostics" / "stellar_masses"
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / "nz_mstar.png"
    else:
        outdir = Path(paths["model_dir"]) / "diagnostics" / "stellar_masses"
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / "nz_mstar.png"

    mstar_cuts_fsf = args.mstar_cuts_fsf if args.mstar_cuts_fsf is not None \
                     else args.mstar_cuts
    make_plot(z, mr, logm_cig, logm_fsf, weights, zbins, vol,
              args.mr_cuts, args.mstar_cuts, mstar_cuts_fsf, outfile)

    logging.info("Done.")


if __name__ == "__main__":
    main()
