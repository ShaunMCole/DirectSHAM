"""
inspect_stellar_masses.py
=========================
Diagnostic plots for the stellar-mass / SFR augmented mock catalogues
produced by add_stellar_masses.py.

Plots produced (saved to <model_dir>/diagnostics/stellar_masses/)
------------------------------------------------------------------
1. cigale_vs_fsf_mstar.png
   Hexbin of CIGALE LOGM vs FSF LOGMSTAR for direct-matched objects,
   with first- and second-pass donor-filled objects overlaid.

2. mr_vs_mstar_cigale.png
   Hexbin of Mr (ABSMAG_RP1) vs CIGALE LOGM, donor-filled overlaid.

3. mr_vs_mstar_fsf.png
   Hexbin of Mr (ABSMAG_RP1) vs FSF LOGMSTAR, donor-filled overlaid.

4. colour_vs_ssfr.png
   Hexbin of rest-frame g-r vs sSFR proxy (CIGALE LOGSFR - LOGM),
   donor-filled overlaid.

5. provenance_redshift.png
   Normalised redshift distributions: all objects vs each provenance
   class (direct, donor1, donor2) for both CIGALE and FSF.

6. mstar_distributions.png
   Normalised stellar-mass distributions in redshift shells
   (0-0.15, 0.15-0.30, 0.30-0.45, 0.45-0.60) for data and mock,
   analogous to the magnitude distribution plot in the main paper.

Usage
-----
python pipeline/inspect_stellar_masses.py --reg NGC SGC
python pipeline/inspect_stellar_masses.py --reg NGC --sig_lgv_ref 0.18 --sig_lgv_slope 0.30
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

# Redshift shells for distribution plots
Z_SHELLS = [(0.00, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.60)]

# Colours for provenance classes
C_DONOR1   = "tab:red"
C_DONOR2   = "tab:cyan"
C_UNRES    = "tab:orange"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_augmented(paths, reg):
    mock_in = Path(paths[f"mock_{reg}"])
    aug     = mock_in.with_name(mock_in.stem + "_augmented.fits")
    if not aug.exists():
        raise FileNotFoundError(f"Augmented mock not found: {aug}\n"
                                "Run add_stellar_masses.py first.")
    logging.info("Reading %s …", aug)
    return Table.read(str(aug))


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(fig, outdir, name, reg):
    fname = outdir / f"{name}_{reg}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("  Saved %s", fname)


def provenance_masks(mock, flag_col):
    f = np.asarray(mock[flag_col], dtype=np.int8)
    return {
        "direct" : f == 1,
        "donor1" : f == 2,
        "donor2" : f == 3,
        "unres"  : f == -1,
    }


# ---------------------------------------------------------------------------
# Plot 1 – CIGALE LOGM vs FSF LOGMSTAR
# ---------------------------------------------------------------------------

def plot_cigale_vs_fsf(mock, outdir, reg):
    pm_cig = provenance_masks(mock, "cigflag")
    matched = pm_cig["direct"]
    d1      = pm_cig["donor1"]
    d2      = pm_cig["donor2"]

    logm     = np.asarray(mock["LOGM"],    dtype=float)
    logmstar = np.asarray(mock["LOGMSTAR"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))

    hb = ax.hexbin(logmstar[matched], logm[matched],
                   gridsize=150, bins="log", mincnt=1, cmap="Blues", zorder=1)
    plt.colorbar(hb, ax=ax, label="N")

    # Donors plotted on top of hexbin (zorder=2) so they are visible
    ax.scatter(logmstar[d1], logm[d1],
               s=1, c=C_DONOR1, alpha=0.05, zorder=2,
               label=f"1st-pass donor ({d1.sum():,})")
    ax.scatter(logmstar[d2], logm[d2],
               s=12, c=C_DONOR2, alpha=1.0, zorder=3,
               label=f"2nd-pass donor ({d2.sum():,})")

    lims = [7, 13]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r"FSF $\log M_\star$")
    ax.set_ylabel(r"CIGALE $\log M_\star$")
    ax.set_title(f"{reg}  —  direct matches: {matched.sum():,}")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    save(fig, outdir, "cigale_vs_fsf_mstar", reg)


# ---------------------------------------------------------------------------
# Plot 2 – Mr vs CIGALE stellar mass
# ---------------------------------------------------------------------------

def plot_mr_vs_mstar(mock, outdir, reg, vac="cigale"):
    col     = "LOGM" if vac == "cigale" else "LOGMSTAR"
    flag    = "cigflag" if vac == "cigale" else "fsfflag"
    ylabel  = (r"CIGALE $\log M_\star$" if vac == "cigale"
               else r"FSF $\log M_\star$")
    pm      = provenance_masks(mock, flag)
    matched = pm["direct"]
    d1      = pm["donor1"]
    d2      = pm["donor2"]

    mr   = np.asarray(mock["ABSMAG_RP1"], dtype=float)
    logm = np.asarray(mock[col],          dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))

    hb = ax.hexbin(mr[matched], logm[matched],
                   gridsize=150, bins="log", mincnt=1, cmap="Blues", zorder=1)
    plt.colorbar(hb, ax=ax, label="N")

    ax.scatter(mr[d1], logm[d1],
               s=1, c=C_DONOR1, alpha=0.05, zorder=2,
               label=f"1st-pass donor ({d1.sum():,})")
    ax.scatter(mr[d2], logm[d2],
               s=12, c=C_DONOR2, alpha=1.0, zorder=3,
               label=f"2nd-pass donor ({d2.sum():,})")

    ax.set_xlabel(r"$M_r - 5\log h$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{reg}  —  direct matches: {matched.sum():,}")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    save(fig, outdir, f"mr_vs_mstar_{vac}", reg)


# ---------------------------------------------------------------------------
# Plot 3 – rest-frame colour vs sSFR proxy
# ---------------------------------------------------------------------------

def plot_colour_vs_ssfr(mock, outdir, reg):
    pm_cig  = provenance_masks(mock, "cigflag")
    matched = pm_cig["direct"]
    d1      = pm_cig["donor1"]
    d2      = pm_cig["donor2"]

    colour   = np.asarray(mock["ABSMAG_GP1"] - mock["ABSMAG_RP1"], dtype=float)
    logm     = np.asarray(mock["LOGM"],   dtype=float)
    logsfr   = np.asarray(mock["LOGSFR"], dtype=float)
    ssfr     = logsfr - logm

    fig, ax = plt.subplots(figsize=(7, 6))

    hb = ax.hexbin(colour[matched], ssfr[matched],
                   gridsize=150, bins="log", mincnt=1, cmap="Blues", zorder=1)
    plt.colorbar(hb, ax=ax, label="N")

    ax.scatter(colour[d1], ssfr[d1],
               s=1, c=C_DONOR1, alpha=0.05, zorder=2,
               label=f"1st-pass donor ({d1.sum():,})")
    ax.scatter(colour[d2], ssfr[d2],
               s=12, c=C_DONOR2, alpha=1.0, zorder=3,
               label=f"2nd-pass donor ({d2.sum():,})")

    ax.set_xlabel(r"$^{0.1}(g-r)$")
    ax.set_ylabel(r"$\log\mathrm{SFR} - \log M_\star$  (CIGALE)")
    ax.set_title(f"{reg}  —  direct matches: {matched.sum():,}")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    save(fig, outdir, "colour_vs_ssfr", reg)


# ---------------------------------------------------------------------------
# Plot 4 – provenance vs redshift
# ---------------------------------------------------------------------------

def plot_provenance_redshift(mock, outdir, reg):
    z = np.asarray(mock["Zobs"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    zbins = np.linspace(0, 0.6, 61)

    for ax, flag_col, title in [
        (axes[0], "cigflag", "CIGALE"),
        (axes[1], "fsfflag", "FSF"),
    ]:
        pm = provenance_masks(mock, flag_col)

        kw = dict(bins=zbins, density=True, histtype="step", lw=1.5)
        MIN_FOR_HIST = 20   # skip classes too small to make a sensible histogram
        ax.hist(z,               label=f"All ({len(mock):,})",             color="black",    **kw)
        ax.hist(z[pm["direct"]], label=f"Direct ({pm['direct'].sum():,})",  color="tab:blue", **kw)
        if pm["donor1"].sum() >= MIN_FOR_HIST:
            ax.hist(z[pm["donor1"]], label=f"Donor-1 ({pm['donor1'].sum():,})",
                    color=C_DONOR1, **kw)
        else:
            ax.scatter(z[pm["donor1"]], np.zeros(pm["donor1"].sum()),
                       s=20, color=C_DONOR1, zorder=3,
                       label=f"Donor-1 ({pm['donor1'].sum():,})")
        if pm["donor2"].sum() >= MIN_FOR_HIST:
            ax.hist(z[pm["donor2"]], label=f"Donor-2 ({pm['donor2'].sum():,})",
                    color=C_DONOR2, **kw)
        else:
            ax.scatter(z[pm["donor2"]], np.zeros(pm["donor2"].sum()),
                       s=30, color=C_DONOR2, marker="^", zorder=4,
                       label=f"Donor-2 ({pm['donor2'].sum():,})")
        if pm["unres"].sum() > 0:
            ax.scatter(z[pm["unres"]], np.zeros(pm["unres"].sum()),
                       s=50, color=C_UNRES, marker="x", zorder=5,
                       label=f"Unresolved ({pm['unres'].sum():,})")

        ax.set_xlabel("Redshift $z$")
        ax.set_ylabel("Normalised density")
        ax.set_title(f"{reg}  —  {title} provenance")
        ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, outdir, "provenance_redshift", reg)


# ---------------------------------------------------------------------------
# Plot 5 – stellar-mass distributions in redshift shells
# ---------------------------------------------------------------------------

def plot_mstar_distributions(mock, outdir, reg):
    """
    Normalised CIGALE log M* distributions in four redshift shells,
    comparing data (mock) with the random catalogue clone distribution.
    Since the random catalogue does not carry stellar masses, we just
    show the mock distribution split by provenance class.
    """
    z    = np.asarray(mock["Zobs"], dtype=float)
    logm = np.asarray(mock["LOGM"], dtype=float)
    pm   = provenance_masks(mock, "cigflag")

    fig, axes = plt.subplots(len(Z_SHELLS), 1,
                             figsize=(6, 3 * len(Z_SHELLS)),
                             sharex=True)
    mbins = np.linspace(7, 13, 61)

    for ax, (zlo, zhi) in zip(axes, Z_SHELLS):
        in_shell = (z >= zlo) & (z < zhi)

        kw = dict(bins=mbins, density=True, histtype="step", lw=1.5)
        ax.hist(logm[in_shell],
                label=f"All ({in_shell.sum():,})", color="black", **kw)
        ax.hist(logm[in_shell & pm["direct"]],
                label="Direct", color="tab:blue", **kw)
        ax.hist(logm[in_shell & pm["donor1"]],
                label="Donor-1", color=C_DONOR1, **kw)
        ax.hist(logm[in_shell & pm["donor2"]],
                label="Donor-2", color=C_DONOR2, **kw)

        ax.set_ylabel("Norm. density")
        ax.text(0.97, 0.95, f"${zlo:.2f} \\leq z < {zhi:.2f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        if ax is axes[0]:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel(r"CIGALE $\log M_\star / M_\odot$")
    fig.suptitle(f"{reg}  —  stellar-mass distributions by redshift shell",
                 fontsize=10)
    fig.tight_layout()
    save(fig, outdir, "mstar_distributions", reg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_region(reg, paths):
    mock   = load_augmented(paths, reg)
    model_dir = Path(paths["model_dir"])
    outdir = ensure_dir(model_dir / "diagnostics" / "stellar_masses")
    logging.info("Output directory: %s", outdir)

    # Diagnostic: verify flag columns look correct after reading from FITS
    for flag_col in ("cigflag", "fsfflag"):
        f = np.asarray(mock[flag_col], dtype=np.int8)
        logging.info("%s: dtype=%s  values=%s  counts: -1:%d 1:%d 2:%d 3:%d",
                     flag_col, mock[flag_col].dtype,
                     np.unique(f),
                     (f==-1).sum(), (f==1).sum(), (f==2).sum(), (f==3).sum())

    logging.info("Plot 1: CIGALE vs FSF stellar mass")
    plot_cigale_vs_fsf(mock, outdir, reg)

    logging.info("Plot 2: Mr vs CIGALE stellar mass")
    plot_mr_vs_mstar(mock, outdir, reg, vac="cigale")

    logging.info("Plot 3: Mr vs FSF stellar mass")
    plot_mr_vs_mstar(mock, outdir, reg, vac="fsf")

    logging.info("Plot 4: colour vs sSFR")
    plot_colour_vs_ssfr(mock, outdir, reg)

    logging.info("Plot 5: provenance vs redshift")
    plot_provenance_redshift(mock, outdir, reg)

    logging.info("Plot 6: stellar-mass distributions in redshift shells")
    plot_mstar_distributions(mock, outdir, reg)

    logging.info("All plots saved to %s", outdir)


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnostic plots for stellar-mass augmented mock catalogues."
    )
    p.add_argument("--reg", nargs="+", default=["NGC", "SGC"],
                   choices=["NGC", "SGC"])
    p.add_argument("--obs_seed",      type=int,   default=0)
    p.add_argument("--sig_lgv_ref",   type=float, default=config.DEFAULT_SIG_LGV_REF)
    p.add_argument("--sig_lgv_slope", type=float, default=config.DEFAULT_SIG_LGV_SLOPE)
    p.add_argument("--f_colour",      type=float, default=config.DEFAULT_F_COLOUR)
    p.add_argument("--alpha",         type=float, default=config.DEFAULT_ALPHA)
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging("inspect_stellar_masses")
    # Attach the configured handlers to the root logger so that
    # module-level logging.info() calls are captured.
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
        process_region(reg, paths)

    logging.info("Done.")


if __name__ == "__main__":
    main()
