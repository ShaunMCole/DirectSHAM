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


# Number of first-pass donor points to show in hexbin plots.
# Kept small so they don't obscure the hexbin; total count shown in legend.
N_DONOR_SUBSAMPLE = 1000


def subsample(mask, n=N_DONOR_SUBSAMPLE, seed=42):
    """Return a boolean mask selecting at most n random objects from mask."""
    idx = np.where(mask)[0]
    if len(idx) <= n:
        return mask
    rng = np.random.default_rng(seed)
    chosen = rng.choice(idx, size=n, replace=False)
    sub = np.zeros(len(mask), dtype=bool)
    sub[chosen] = True
    return sub


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

def running_stats(x, y, bins=40, plo=5, phi=95, x_min=None, x_max=None):
    """
    Compute running median and percentiles of y in bins of x.
    Returns (bin_centres, medians, p_lo, p_hi) with NaN where bins are empty.
    """
    xlo = max(np.nanpercentile(x, 0.01), x_min) if x_min is not None \
          else np.nanpercentile(x, 0.01)
    xhi = min(np.nanpercentile(x, 99.9), x_max) if x_max is not None \
          else np.nanpercentile(x, 99.9)
    edges  = np.linspace(xlo, xhi, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    med  = np.full(bins, np.nan)
    qlo  = np.full(bins, np.nan)
    qhi  = np.full(bins, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= lo) & (x < hi) & np.isfinite(y)
        if mask.sum() > 5:
            med[i] = np.median(y[mask])
            qlo[i] = np.percentile(y[mask], plo)
            qhi[i] = np.percentile(y[mask], phi)
    return centres, med, qlo, qhi


def plot_cigale_vs_fsf(mock, outdir, reg):
    pm_cig  = provenance_masks(mock, "cigflag")
    matched = pm_cig["direct"]
    d1      = pm_cig["donor1"]
    d2      = pm_cig["donor2"]

    logm     = np.asarray(mock["LOGM"],    dtype=float)
    logmstar = np.asarray(mock["LOGMSTAR"], dtype=float)

    # Only use finite values for statistics
    ok = np.isfinite(logm) & np.isfinite(logmstar)

    # MNRAS single-column figure: 84 mm wide ~ 3.31 inches
    FSIZE  = 7    # base font size
    FSIZE_S = 6   # small font (legend, colorbar)
    plt.rcParams.update({"font.size": FSIZE, "axes.labelsize": FSIZE,
                         "xtick.labelsize": FSIZE_S, "ytick.labelsize": FSIZE_S})

    fig, ax = plt.subplots(figsize=(3.31, 3.0))

    hb = ax.hexbin(logmstar[matched & ok], logm[matched & ok],
                   gridsize=100, bins="log", mincnt=1, cmap="Blues", zorder=1)
    cb = plt.colorbar(hb, ax=ax, label="N")
    cb.ax.tick_params(labelsize=FSIZE_S)
    cb.set_label("N", fontsize=FSIZE_S)

    # Donors plotted on top of hexbin — subsample d1 to avoid obscuring hexbin
    d1_sub = subsample(d1 & ok)
    ax.scatter(logmstar[d1_sub], logm[d1_sub],
               s=0.3, c=C_DONOR1, alpha=0.6, zorder=2,
               label=f"1st-pass donor ({(d1 & ok).sum():,})")
    ax.scatter(logmstar[d2 & ok], logm[d2 & ok],
               s=3, c=C_DONOR2, alpha=1.0, zorder=3,
               label=f"2nd-pass donor ({(d2 & ok).sum():,})")

    lims   = [7, 13]
    offset = -0.35   # known model offset: FSF ~0.35 dex above CIGALE

    # 1:1 line
    ax.plot(lims, lims, "k--", lw=0.6, alpha=0.5, zorder=4)

    # Offset line (1:1 shifted down by 0.35 dex)
    ax.plot(lims, [l + offset for l in lims], "k:", lw=0.8, alpha=0.7,
            zorder=4, label=rf"1:1 $-$ {abs(offset):.2f} dex")

    # Running median and percentiles of CIGALE in bins of FSF
    xdata = logmstar[ok]
    ydata = logm[ok]
    bx, med, qlo, qhi = running_stats(xdata, ydata, x_max=12.3, x_min=7.0)
    valid = np.isfinite(med)
    ax.plot(bx[valid], med[valid], "r-", lw=1.0, zorder=5, label="Median")
    ax.fill_between(bx[valid], qlo[valid], qhi[valid],
                    color="red", alpha=0.2, zorder=4,
                    label=r"5$^{\rm th}$–95$^{\rm th}$ percentile")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r"FSF $\log(M_\star/\mathrm{M}_\odot)$")
    ax.set_ylabel(r"CIGALE $\log(M_\star/\mathrm{M}_\odot)$")
    # No title — caption carries this information
    ax.legend(fontsize=FSIZE_S, markerscale=2)
    fig.tight_layout(pad=0.5)
    save(fig, outdir, "cigale_vs_fsf_mstar", reg)

    # Reset rcParams to defaults so other plots in the same run are unaffected
    plt.rcdefaults()


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

    d1_sub = subsample(d1)
    ax.scatter(mr[d1_sub], logm[d1_sub],
               s=1, c=C_DONOR1, alpha=1.0, zorder=2,
               label=f"1st-pass donor ({d1.sum():,}, showing {d1_sub.sum()})")
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
# Plot 3b – CIGALE log SFR vs FSF log SFR
# ---------------------------------------------------------------------------

def plot_cigale_vs_fsf_sfr(mock, outdir, reg):
    """
    Plot log10 sSFR (CIGALE) vs log10 sSFR (FSF).

    CIGALE:  log10 sSFR = LOGSFR - LOGM          [both already log10]
    FSF:     log10 sSFR = log10(SFR) - LOGMSTAR  [SFR is linear, LOGMSTAR is log10]
    Units: yr^-1
    """
    pm_cig  = provenance_masks(mock, "cigflag")
    matched = pm_cig["direct"]
    d1      = pm_cig["donor1"]
    d2      = pm_cig["donor2"]

    # CIGALE log10 sSFR
    logsfr_cig = np.asarray(mock["LOGSFR"], dtype=float)
    logm_cig   = np.asarray(mock["LOGM"],   dtype=float)
    logssfr_cig = logsfr_cig - logm_cig

    # FSF log10 sSFR
    sfr_fsf    = np.asarray(mock["SFR"],     dtype=float)
    logm_fsf   = np.asarray(mock["LOGMSTAR"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logsfr_fsf = np.where(sfr_fsf > 0, np.log10(sfr_fsf), np.nan)
    logssfr_fsf = logsfr_fsf - logm_fsf

    # Only plot rows where both are finite
    ok = np.isfinite(logssfr_cig) & np.isfinite(logssfr_fsf)

    lims    = [-14, -7]
    ssfr_cut = -11.0   # CIGALE passive/SF boundary (M. Siudek)
    bins_marg = np.linspace(lims[0], lims[1], 120)

    # Layout: main hexbin + right marginal + top marginal
    fig = plt.figure(figsize=(8, 7))
    gs  = fig.add_gridspec(2, 2,
                           width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.04, wspace=0.04)
    ax      = fig.add_subplot(gs[1, 0])   # main panel
    ax_top  = fig.add_subplot(gs[0, 0], sharex=ax)  # FSF marginal (top)
    ax_rgt  = fig.add_subplot(gs[1, 1], sharey=ax)  # CIGALE marginal (right)

    # --- Main hexbin ---
    hb = ax.hexbin(logssfr_fsf[ok & matched], logssfr_cig[ok & matched],
                   gridsize=120, bins="log", mincnt=1, cmap="Blues", zorder=1)
    cb = fig.colorbar(hb, ax=ax_rgt, label="N", fraction=0.15, pad=0.04)
    cb.ax.tick_params(labelsize=8)

    d1_sub = subsample(ok & d1)
    ax.scatter(logssfr_fsf[d1_sub], logssfr_cig[d1_sub],
               s=1, c=C_DONOR1, alpha=1.0, zorder=2,
               label=f"1st-pass donor ({(ok & d1).sum():,}, showing {d1_sub.sum()})")
    ax.scatter(logssfr_fsf[ok & d2], logssfr_cig[ok & d2],
               s=12, c=C_DONOR2, alpha=1.0, zorder=3,
               label=f"2nd-pass donor ({(ok & d2).sum():,})")

    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)

    # CIGALE SF/Q cut line (horizontal)
    ax.axhline(ssfr_cut, color="tab:orange", lw=1.2, ls="--", zorder=5,
               label=rf"CIGALE cut $\log\,\mathrm{{sSFR}}={ssfr_cut}$")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(r"FSF $\log_{10}\,\mathrm{sSFR}\ [\mathrm{yr}^{-1}]$")
    ax.set_ylabel(r"CIGALE $\log_{10}\,\mathrm{sSFR}\ [\mathrm{yr}^{-1}]$")
    ax.legend(fontsize=7, markerscale=2, loc="upper left")

    # --- Top marginal: FSF sSFR ---
    kw = dict(bins=bins_marg, density=True, histtype="step", lw=1.2)
    ax_top.hist(logssfr_fsf[ok & matched], color="steelblue",
                label="FSF", **kw)
    ax_top.set_ylabel("Density", fontsize=8)
    ax_top.tick_params(labelbottom=False, labelsize=7)
    ax_top.set_xlim(lims)

    # --- Right marginal: CIGALE sSFR ---
    ax_rgt.hist(logssfr_cig[ok & matched], bins=bins_marg, density=True,
                histtype="step", lw=1.2, color="tab:red",
                orientation="horizontal", label="CIGALE")
    # Show the cut on the marginal too
    ax_rgt.axhline(ssfr_cut, color="tab:orange", lw=1.2, ls="--")
    ax_rgt.tick_params(labelleft=False, labelsize=7)
    ax_rgt.set_xlabel("Density", fontsize=8)
    ax_rgt.set_ylim(lims)

    fig.suptitle(f"{reg}  —  direct matches: {(ok & matched).sum():,}  "
                 f"(finite sSFR in both)", fontsize=9)
    fig.tight_layout()
    save(fig, outdir, "cigale_vs_fsf_ssfr", reg)


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

    d1_sub = subsample(d1)
    ax.scatter(colour[d1_sub], ssfr[d1_sub],
               s=1, c=C_DONOR1, alpha=1.0, zorder=2,
               label=f"1st-pass donor ({d1.sum():,}, showing {d1_sub.sum()})")
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

    logging.info("Plot 3b: CIGALE vs FSF sSFR")
    plot_cigale_vs_fsf_sfr(mock, outdir, reg)

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
    p.add_argument("--cat", type=Path, default=None,
                   help="Read from this catalogue instead of the mock "
                        "(e.g. BGS_BRIGHT_6thOct_augmented.fits). "
                        "Produces only the CIGALE vs FSF plot. "
                        "The --reg argument is ignored.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Override output directory for plots")
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging("inspect_stellar_masses")
    root = logging.getLogger()
    root.setLevel(log.level)
    for h in log.handlers:
        root.addHandler(h)

    # --cat mode: produce only the CIGALE vs FSF plot from a master catalogue
    if args.cat is not None:
        if not args.cat.exists():
            logging.error("Catalogue not found: %s", args.cat)
            sys.exit(1)
        logging.info("Reading catalogue: %s", args.cat)
        cat = Table.read(str(args.cat))
        logging.info("  %d rows", len(cat))

        outdir = ensure_dir(args.outdir) if args.outdir else \
                 ensure_dir(args.cat.parent / "diagnostics" / "stellar_masses")
        logging.info("Output directory: %s", outdir)

        # Use filename stem as region label (e.g. "BGS_BRIGHT_6thOct_augmented")
        reg = args.cat.stem
        plot_cigale_vs_fsf(cat, outdir, reg)
        logging.info("Done.")
        return

    # Standard mock mode
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
