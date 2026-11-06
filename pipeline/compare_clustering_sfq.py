"""
compare_clustering_sfq.py — Five-panel passive/star-forming clustering figure
for the MNRAS paper, single-column vertical layout with shared x-axis.
Styled to match S5PanelColourClustering2.py.

Each panel shows s²ξ₀ and s²ξ₂ for BGS DR2 data (with jackknife errors)
and two mock lines: quenched/passive (red) and star-forming (blue), split by
CIGALE log(sSFR) = -11.

Run from ~/DirectSHAM/:
    python pipeline/compare_clustering_sfq.py \\
        --sig_lgv_ref 0.18 --sig_lgv_slope 0.30
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycorr import TwoPointCorrelationFunction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import make_paths, setup_logging
from config import (DATA_DATE_DIR,
                    DEFAULT_OBS_SEED, DEFAULT_SIG_LGV_REF,
                    DEFAULT_SIG_LGV_SLOPE, DEFAULT_F_COLOUR, DEFAULT_ALPHA)

# =============================================================================
# Samples
# =============================================================================

SAMPLES = [
    dict(cig_cut=10.60, zmin=0.02, zmax=0.34),
    dict(cig_cut=10.10, zmin=0.02, zmax=0.27),
    dict(cig_cut= 9.50, zmin=0.02, zmax=0.21),
    dict(cig_cut= 8.90, zmin=0.02, zmax=0.17),
]

YMAX = {10.60: 119.99, 10.10: 99.99, 9.50: 99.99, 8.90: 99.99}

# Colour palette matching S5PanelColourClustering2 convention
DATA_COLOURS = {
    'passive': {'xi0': '#8B0000', 'xi2': '#FF6B6B'},
    'sf':      {'xi0': '#00008B', 'xi2': '#6B9FFF'},
}
DATA_MARKERS = {'xi0': 'o', 'xi2': 's'}
MOCK_LS      = {'xi0': '-',  'xi2': '--'}

DATA_NJACK = 60

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obs_seed",      type=int,   default=DEFAULT_OBS_SEED)
    p.add_argument("--sig_lgv_ref",   type=float, default=DEFAULT_SIG_LGV_REF)
    p.add_argument("--sig_lgv_slope", type=float, default=DEFAULT_SIG_LGV_SLOPE)
    p.add_argument("--f_colour",      type=float, default=DEFAULT_F_COLOUR)
    p.add_argument("--alpha",         type=float, default=DEFAULT_ALPHA)
    p.add_argument("--smin",          type=float, default=5.0)
    p.add_argument("--smax",          type=float, default=50.0)
    p.add_argument("--savefig",       type=str,   default=None)
    return p.parse_args()

# =============================================================================
# Path helpers
# =============================================================================

def data_npy(cig_cut, zmin, zmax, sfq):
    tag = f"Mcig{cig_cut}_z{zmin:.2f}-{zmax:.2f}_{sfq}"
    smu = DATA_DATE_DIR / "clustering" / "data" / tag / "smu"
    pat = (f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.2f}_{zmax:.2f}"
           f"_pip_angular_bitwise_log_njack{DATA_NJACK}_nran1_split20.npy")
    return smu / pat

def mock_npy(cig_cut, zmin, zmax, sfq, paths):
    tag = f"Mcig{cig_cut}_z{zmin:.2f}-{zmax:.2f}_{sfq}"
    smu = Path(paths["model_dir"]) / "clustering" / tag / "smu"
    pat = (f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.2f}_{zmax:.2f}"
           f"_default_log_njack0_nran1_split20.npy")
    return smu / pat

# =============================================================================
# Load / extract / chi2  (identical to S5PanelColourClustering2)
# =============================================================================

def load_result(path):
    path = Path(path)
    if not path.exists():
        print(f"WARNING: File not found: {path}")
        return None
    return TwoPointCorrelationFunction.load(str(path))

def extract_multipoles(result, ells=(0, 2)):
    s = result.sep
    if s.ndim == 2:
        s = np.nanmean(s, axis=1)
    try:
        xis, cov = result.get_corr(ells=tuple(ells), return_cov=True)
        xis  = np.atleast_2d(xis)
        ns   = xis.shape[1]
        errs = np.sqrt(np.diag(cov))
        err_dict = {ell: errs[i*ns:(i+1)*ns] for i, ell in enumerate(ells)}
        has_cov  = True
    except Exception:
        xis      = np.atleast_2d(result(ells=tuple(ells)))
        ns       = xis.shape[1]
        err_dict = {ell: np.zeros(ns) for ell in ells}
        has_cov  = False
    xi_dict = {ell: xis[i] for i, ell in enumerate(ells)}
    good = np.isfinite(s) & (s > 0)
    for xi in xi_dict.values():
        good &= np.isfinite(xi)
    return s, xi_dict, err_dict, good, has_cov

def compute_chi2(s_dat, xi_dat, err_dat, s_mock, xi_mock, ells,
                 smin=5.0, smax=50.0):
    fit   = (s_dat >= smin) & (s_dat <= smax)
    s_fit = s_dat[fit]
    ndof  = int(fit.sum()) * len(ells)
    chi2  = 0.0
    for ell in ells:
        xi_d = xi_dat[ell][fit]
        xi_m = np.interp(s_fit, s_mock, xi_mock[ell])
        err  = err_dat[ell][fit]
        err  = np.where((err > 0) & np.isfinite(err), err, 1.0)
        chi2 += float(np.sum(((xi_d - xi_m) / err)**2))
    return chi2, ndof

# =============================================================================
# Main
# =============================================================================

def main():
    args   = parse_args()
    log    = setup_logging("compare_clustering_sfq")
    ells   = [0, 2]
    splits = ['passive', 'sf']

    paths = make_paths(obs_seed=args.obs_seed,
                       sig_lgv_ref=args.sig_lgv_ref,
                       sig_lgv_slope=args.sig_lgv_slope,
                       f_colour=args.f_colour, alpha=args.alpha)

    n   = len(SAMPLES)
    fig, axes = plt.subplots(n, 1, figsize=(3.35, 2.0 * n),
                             sharex=True,
                             gridspec_kw=dict(hspace=0))

    for i, (sample, ax) in enumerate(zip(SAMPLES, axes)):
        cig_cut = sample["cig_cut"]
        zmin    = sample["zmin"]
        zmax    = sample["zmax"]

        ymax = YMAX.get(cig_cut, 100)
        ymin = -ymax * 0.5

        data_handles = []
        mock_handles = []

        for sfq in splits:
            # --- Data ---
            data_result = load_result(data_npy(cig_cut, zmin, zmax, sfq))
            if data_result is not None:
                s_dat, xi_dat, err_dat, good_dat, has_cov = extract_multipoles(
                    data_result, ells)
                s_d       = s_dat[good_dat]
                xi_dat_g  = {ell: xi_dat[ell][good_dat] for ell in ells}
                err_dat_g = {ell: err_dat[ell][good_dat] for ell in ells}

                for ell in ells:
                    c    = DATA_COLOURS[sfq][f'xi{ell}']
                    fmt  = DATA_MARKERS[f'xi{ell}']
                    y    = s_d**2 * xi_dat_g[ell]
                    yerr = s_d**2 * err_dat_g[ell]
                    lbl  = ('Quenched' if sfq == 'passive' else 'Star-forming') + \
                           rf' $\xi_{ell}$'
                    if has_cov and np.any(err_dat_g[ell] > 0):
                        h = ax.errorbar(s_d, y, yerr=yerr,
                                        fmt=fmt, color=c,
                                        ms=1.5, capsize=1.5, lw=0.7,
                                        zorder=10, label=lbl)
                    else:
                        h, = ax.plot(s_d, y, fmt, color=c, ms=1.5,
                                     zorder=10, label=lbl)
                    if ell == 0:
                        data_handles.append(h)

            # --- Mock ---
            mock_result = load_result(mock_npy(cig_cut, zmin, zmax,
                                               sfq, paths))
            if mock_result is not None:
                s_m, xi_m, _, good_m, _ = extract_multipoles(mock_result, ells)
                xi_m_g = {ell: xi_m[ell][good_m] for ell in ells}
                s_ = s_m[good_m]

                # chi2 vs data
                if data_result is not None:
                    chi2, ndof = compute_chi2(
                        s_d, xi_dat_g, err_dat_g,
                        s_, xi_m_g, ells, args.smin, args.smax)
                    log.info("  Mcig%.2f %s  chi2/df=%.2f (%d)",
                             cig_cut, sfq, chi2/ndof, ndof)

                line_col = DATA_COLOURS[sfq]['xi0']
                mock_lbl = 'Quenched mock' if sfq == 'passive' else 'Star-forming mock'
                for ell in ells:
                    h, = ax.plot(s_, s_**2 * xi_m_g[ell],
                                 lw=0.7, color=line_col,
                                 ls=MOCK_LS[f'xi{ell}'],
                                 label=mock_lbl if ell == 0 else '_nolegend_')
                    if ell == 0:
                        mock_handles.append(h)

        # Sample label — bottom left
        ax.text(0.03, 0.04,
                rf"$\log M_\star>{cig_cut:.2f}$,  "
                rf"${zmin:.2f}\leq z\leq{zmax:.2f}$",
                transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          alpha=0.7, ec='none'))

        # Data legend panel 0, mock legend panel 1
        if i == 0:
            ax.legend(handles=data_handles, fontsize=7,
                      loc='upper left', framealpha=0.8)
        if i == 1:
            ax.legend(handles=mock_handles, fontsize=7,
                      loc='upper left', framealpha=0.8)

        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.set_xscale('log')
        ax.set_xlim(0.1, 80)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(r'$s^2\xi_\ell(s)$ [Mpc$^2h^{-2}$]', fontsize=9)
        ax.tick_params(labelsize=8, which='both', top=True, right=False)
        if i < n - 1:
            ax.tick_params(labelbottom=False, which='both')

    axes[-1].set_xlabel(r'$s$ [Mpc/$h$]', fontsize=9)
    axes[0].tick_params(which='both', top=False)

    if args.savefig:
        outpath = args.savefig
    else:
        model_tag = (f"seed{args.obs_seed:03d}_sig_lgv"
                     f"{args.sig_lgv_ref:.2f}_{args.sig_lgv_slope:.2f}"
                     f"_fc{args.f_colour:.2f}_al{args.alpha:.2f}")
        outdir = (DATA_DATE_DIR / "N_neigh_lum50" / model_tag
                  / "diagnostics" / "stellar_masses")
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = str(outdir / "clustering_sfq_5panel")

    for ext in (".pdf", ".jpg"):
        fig.savefig(outpath + ext,
                    dpi=150 if ext == ".jpg" else None,
                    bbox_inches="tight")
        log.info("Saved: %s%s", outpath, ext)

    plt.close(fig)


if __name__ == "__main__":
    main()
