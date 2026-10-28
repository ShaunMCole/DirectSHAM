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


def load_data(paths, reg):
    mock_in = Path(paths[f"mock_{reg}"])
    aug = mock_in.with_name(mock_in.stem + "_augmented.fits")
    if not aug.exists():
        raise FileNotFoundError(f"Augmented mock not found: {aug}")
    logging.info("Reading %s …", aug)
    cols = ["Zobs", "ABSMAG_RP1", "LOGM", "LOGMSTAR", "WEIGHT"]
    t = Table.read(str(aug), format="fits")
    cols = [c for c in cols if c in t.colnames]
    return t[cols]


def nz_line(ax, z, vals, threshold, weights, zbins, vol, color, label):
    """Plot n(z) as a line for objects with vals >= threshold."""
    mask = np.isfinite(vals) & (vals >= threshold)
    h, _ = np.histogram(z[mask], bins=zbins,
                        weights=weights[mask] if weights is not None else None)
    nz = np.log10(h / vol + EPS)
    zcen = 0.5 * (zbins[:-1] + zbins[1:])
    ax.plot(zcen, nz, color=color, lw=1.5, label=label)


def nz_scatter_mr(ax, z, mr, threshold, weights, zbins, vol, color, label):
    """Plot n(z) scatter points for objects with mr < threshold (Mr is negative)."""
    mask = np.isfinite(mr) & (mr < threshold)
    h, _ = np.histogram(z[mask], bins=zbins,
                        weights=weights[mask] if weights is not None else None)
    nz = np.log10(h / vol + EPS)
    zcen = 0.5 * (zbins[:-1] + zbins[1:])
    ax.scatter(zcen, nz, s=10, color=color, label=label)


def make_plot(z, mr, logm_cig, logm_fsf, weights, zbins, vol,
              mr_cuts, mstar_cuts, mstar_cuts_fsf, outfile):

    mr_colors  = plt.cm.Blues_r( np.linspace(0.2, 0.7, len(mr_cuts)))
    cig_colors = plt.cm.Reds_r(  np.linspace(0.2, 0.7, len(mstar_cuts)))
    fsf_colors = plt.cm.Greens_r(np.linspace(0.2, 0.7, len(mstar_cuts_fsf)))

    xlim = (Z_MIN, 0.50)
    ylim = (-4.0, -1.25)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Mr thresholds — blue scatter
    for cut, col in zip(mr_cuts, mr_colors):
        nz_scatter_mr(ax, z, mr, cut, weights, zbins, vol, col,
                      label=rf"$M_r < {cut:.1f}$")

    # CIGALE M* thresholds — red lines
    for cut, col in zip(mstar_cuts, cig_colors):
        nz_line(ax, z, logm_cig, cut, weights, zbins, vol, col,
                label=rf"CIGALE $\log M_* > {cut:.2f}$")

    # FSF M* thresholds — green lines
    for cut, col in zip(mstar_cuts_fsf, fsf_colors):
        nz_line(ax, z, logm_fsf, cut, weights, zbins, vol, col,
                label=rf"FSF $\log M_* > {cut:.2f}$")

    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$\log_{10}\,\bar{n}(z)\ [h^3\,\mathrm{Mpc}^{-3}]$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    ax.set_title(r"BGS data: $\bar{n}(z)$ by magnitude and stellar mass threshold")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
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

    # Load and combine NGC + SGC
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

    z        = np.asarray(data["Zobs"],       dtype=float)
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
        for reg in ("NGC", "SGC"):
            ran_path = Path(paths[f"formock_ran_{reg}"])
            if ran_path.exists():
                hdr = getheader(str(ran_path), ext=1)
                fsky += hdr.get("FSKY", 0.0)
            else:
                logging.warning("Formock random not found: %s", ran_path)
        logging.info("FSKY from formock headers: %.6f", fsky)

    zbins = np.linspace(Z_MIN, Z_MAX, N_BINS + 1)
    dc    = cosmo.comoving_distance(zbins).value   # Mpc/h
    vol   = (4.0 * np.pi * fsky / 3.0) * (dc[1:]**3 - dc[:-1]**3)

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
