"""
compare_clustering_mstar.py — Five-panel stellar mass threshold clustering
figure for the MNRAS paper, single-column vertical layout with shared x-axis.

Each panel shows s²ξ₀ and s²ξ₂ for BGS DR2 data (with jackknife errors)
and two mock lines: CIGALE threshold (red) and FSF threshold (green).

Run from ~/DirectSHAM/:
    python pipeline/compare_clustering_mstar.py \\
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
    dict(cig_cut=11.10, fsf_cut=11.45, zmin=0.02, zmax=0.42),
    dict(cig_cut=10.60, fsf_cut=10.95, zmin=0.02, zmax=0.34),
    dict(cig_cut=10.10, fsf_cut=10.45, zmin=0.02, zmax=0.27),
    dict(cig_cut= 9.50, fsf_cut= 9.85, zmin=0.02, zmax=0.21),
    dict(cig_cut= 8.90, fsf_cut= 9.25, zmin=0.02, zmax=0.17),
]

YMAX = {11.10: 150, 10.60: 120, 10.10: 99.99, 9.50: 99.99, 8.90: 99.99}

# Colours matching S5PanelClustering2 palette
COL_XI0_DATA = '#1f1f1f'
COL_XI2_DATA = '#888888'
COL_CIG_XI0  = '#c0392b'   # dark red
COL_CIG_XI2  = '#e67e73'   # light red
COL_FSF_XI0  = '#27ae60'   # dark green
COL_FSF_XI2  = '#82c99e'   # light green

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

def data_npy(sample_type, cut, zmin, zmax):
    tag = f"M{sample_type}{cut}_z{zmin:.2f}-{zmax:.2f}"
    smu = DATA_DATE_DIR / "clustering" / "data" / tag / "smu"
    pat = (f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.2f}_{zmax:.2f}"
           f"_pip_angular_bitwise_log_njack{DATA_NJACK}_nran1_split20.npy")
    return smu / pat

def mock_npy(sample_type, cut, zmin, zmax, paths):
    tag = f"M{sample_type}{cut}_z{zmin:.2f}-{zmax:.2f}"
    smu = Path(paths["model_dir"]) / "clustering" / tag / "smu"
    pat = (f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.2f}_{zmax:.2f}"
           f"_default_log_njack0_nran1_split20.npy")
    return smu / pat

# =============================================================================
# Load / extract (identical to S5PanelClustering2)
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
    args = parse_args()
    log  = setup_logging("compare_clustering_mstar")
    ells = [0, 2]

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
        fsf_cut = sample["fsf_cut"]
        zmin    = sample["zmin"]
        zmax    = sample["zmax"]

        ymax = YMAX.get(cig_cut, 100)
        ymin = -ymax * 0.5

        # ------------------------------------------------------------------
        # Data (CIGALE selection)
        # ------------------------------------------------------------------
        data_result = load_result(data_npy("cig", cig_cut, zmin, zmax))
        legend_handles = []
        s_d = xi_dat_g = err_dat_g = None

        if data_result is not None:
            s_dat, xi_dat, err_dat, good_dat, has_cov = extract_multipoles(
                data_result, ells)
            s_d       = s_dat[good_dat]
            xi_dat_g  = {ell: xi_dat[ell][good_dat]  for ell in ells}
            err_dat_g = {ell: err_dat[ell][good_dat]  for ell in ells}

            for ell, fmt, col, lbl in [
                (0, 'o', COL_XI0_DATA, r'Data $\xi_0$'),
                (2, 's', COL_XI2_DATA, r'Data $\xi_2$'),
            ]:
                y    = s_d**2 * xi_dat_g[ell]
                yerr = s_d**2 * err_dat_g[ell]
                if has_cov and np.any(err_dat_g[ell] > 0):
                    h = ax.errorbar(s_d, y, yerr=yerr, fmt=fmt, color=col,
                                    ms=1.5, capsize=1.5, lw=0.7, zorder=10,
                                    label=lbl)
                else:
                    h, = ax.plot(s_d, y, fmt, color=col, ms=1.5,
                                 zorder=10, label=lbl)
                legend_handles.append(h)

        # ------------------------------------------------------------------
        # Mocks: CIGALE (red) then FSF (green)
        # ------------------------------------------------------------------
        for sample_type, cut, col0, col2, vac_lbl in [
            ("cig", cig_cut, COL_CIG_XI0, COL_CIG_XI2,
             f"CIGALE ${cig_cut:.2f}$"),
            ("fsf", fsf_cut, COL_FSF_XI0, COL_FSF_XI2,
             f"FSF ${fsf_cut:.2f}$"),
        ]:
            mock_result = load_result(mock_npy(sample_type, cut,
                                               zmin, zmax, paths))
            if mock_result is None:
                continue

            s_m, xi_m, _, good_m, _ = extract_multipoles(mock_result, ells)
            xi_m_g = {ell: xi_m[ell][good_m] for ell in ells}

            chi2_str = "?"
            if s_d is not None:
                chi2, ndof = compute_chi2(s_d, xi_dat_g, err_dat_g,
                                          s_m[good_m], xi_m_g, ells,
                                          args.smin, args.smax)
                chi2_str = f"{chi2/ndof:.1f}"
                log.info("  %s  chi2/df=%s (%d)", vac_lbl, chi2_str, ndof)

            for ell, col, ls in [(0, col0, '-'), (2, col2, '--')]:
                lbl = vac_lbl if ls == '-' else None
                h, = ax.plot(s_m[good_m], s_m[good_m]**2 * xi_m_g[ell],
                             lw=0.8, color=col, ls=ls, label=lbl, zorder=5)
                if ls == '-':
                    legend_handles.append(h)

        # ------------------------------------------------------------------
        # Sample label — bottom left
        # ------------------------------------------------------------------
        ax.text(0.03, 0.04,
                rf"$\log M_\star>{cig_cut:.2f}$,  "
                rf"${zmin:.2f}\leq z\leq{zmax:.2f}$",
                transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          alpha=0.7, ec='none'))

        # Curve legend — top panel only, all handles
        if i == 0:
            ax.legend(handles=legend_handles, fontsize=7,
                      loc='upper left', framealpha=0.8)

        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.set_xscale('log')
        ax.set_xlim(0.1, 80)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(r'$s^2\xi_\ell(s)$ [Mpc$^2h^{-2}$]', fontsize=9)
        ax.tick_params(labelsize=8, which='both', top=True, right=False)
        if i < n - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel(r'$s$ [Mpc/$h$]', fontsize=9)
    axes[0].tick_params(which='both', top=False)

    # Output path
    if args.savefig:
        outpath = args.savefig
    else:
        model_tag = (f"seed{args.obs_seed:03d}_sig_lgv"
                     f"{args.sig_lgv_ref:.2f}_{args.sig_lgv_slope:.2f}"
                     f"_fc{args.f_colour:.2f}_al{args.alpha:.2f}")
        outdir = (DATA_DATE_DIR / "N_neigh_lum50" / model_tag
                  / "diagnostics" / "stellar_masses")
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = str(outdir / "clustering_mstar_5panel")

    for ext in (".pdf", ".jpg"):
        fig.savefig(outpath + ext,
                    dpi=150 if ext == ".jpg" else None,
                    bbox_inches="tight")
        log.info("Saved: %s%s", outpath, ext)

    plt.close(fig)


if __name__ == "__main__":
    main()
