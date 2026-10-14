"""
compare_clustering_5panel.py — Five-panel comparison of BGS data and mock
two-point correlation function multipoles for the standard magnitude-limited
samples.

Layout (following Elena et al.):
    [ Mr<-22.0 ]  [ Mr<-21.5 ]
    [ Mr<-21.0 ]  [ Mr<-20.5 ]
    [ Mr<-20.0 ]

Each panel shows s²ξ₀ and s²ξ₂ for data (with jackknife error bars) and one
or more mocks.

Usage examples
--------------
# Default: one mock, sig_lgv=0.04, obs_seed=0
python compare_clustering_5panel.py

# Specify mock parameters explicitly:
python compare_clustering_5panel.py --sig_lgv 0.04 --obs_seed 0

# Compare two scatter values:
python compare_clustering_5panel.py --sig_lgv 0.04 0.06

# Save the figure:
python compare_clustering_5panel.py --savefig 5panel_comparison.png
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import setup_logging
from config import DATA_DIR

# =============================================================================
# The 5 standard samples
# =============================================================================

SAMPLES = [
    dict(mr_cut=-22.0, zmin=0.02, zmax=0.50),
    dict(mr_cut=-21.5, zmin=0.02, zmax=0.42),
    dict(mr_cut=-21.0, zmin=0.02, zmax=0.34),
    dict(mr_cut=-20.5, zmin=0.02, zmax=0.28),
    dict(mr_cut=-20.0, zmin=0.02, zmax=0.23),
]

# Panel layout: list of (row, col) positions, 2 columns
LAYOUT = [(0,0), (0,1), (1,0), (1,1), (2,0)]


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sig_lgv", nargs="+", type=float, default=[0.04],
        help="One or more log10(Vpeak) scatter values to compare (default: 0.04)",
    )
    p.add_argument(
        "--obs_seed", type=int, default=0,
        help="Observer position seed (default: 0)",
    )
    p.add_argument(
        "--use_jackknife", action="store_true",
        help="Use jackknife diagonal errors for chi2.",
    )
    p.add_argument(
        "--smin", type=float, default=5.0,
        help="Minimum separation [Mpc/h] for chi2 (default: 5.0)",
    )
    p.add_argument(
        "--smax", type=float, default=50.0,
        help="Maximum separation [Mpc/h] for chi2 (default: 50.0)",
    )
    p.add_argument(
        "--xlim", nargs=2, type=float, default=[0.1, 80],
        help="x-axis limits (default: 0.1 80)",
    )
    p.add_argument(
        "--savefig", type=str, default=None,
        help="Path to save the figure (default: display interactively).",
    )
    p.add_argument(
        "--data_njack", type=int, default=60,
        help="njack value used when measuring data clustering (default: 60)",
    )
    return p.parse_args()


# =============================================================================
# Path helpers
# =============================================================================

def data_file(mr_cut, zmin, zmax, njack):
    sample_tag = f"Mr{mr_cut}_z{zmin}-{zmax}"
    smu_dir    = DATA_DIR / "clustering" / "data" / sample_tag / "smu"
    pattern    = f"allcounts_BGS_BRIGHT_GCcomb_{zmin}_{zmax}_pip_angular_bitwise_log_njack{njack}_nran1_split20.npy"
    return smu_dir / pattern


def mock_file(mr_cut, zmin, zmax, obs_seed, sig_lgv):
    sample_tag = f"Mr{mr_cut}_z{zmin}-{zmax}"
    mock_tag   = f"seed{obs_seed:03d}_sig_lgv{sig_lgv:.2f}"
    smu_dir    = DATA_DIR / "clustering" / "mocks" / mock_tag / sample_tag / "smu"
    pattern    = f"allcounts_BGS_BRIGHT_GCcomb_{zmin}_{zmax}_default_log_njack0_nran1_split20.npy"
    return smu_dir / pattern


# =============================================================================
# Loading and extraction
# =============================================================================

def load_result(path, log):
    from pycorr import TwoPointCorrelationFunction
    path = Path(path)
    if not path.exists():
        log.warning(f"File not found: {path}")
        return None
    result = TwoPointCorrelationFunction.load(str(path))
    log.info(f"Loaded: {path.name}")
    return result


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
    good    = np.isfinite(s) & (s > 0)
    for xi in xi_dict.values():
        good &= np.isfinite(xi)
    return s, xi_dict, err_dict, good, has_cov


def compute_chi2(s_dat, xi_dat, err_dat, s_mock, xi_mock,
                 ells, smin, smax, use_jackknife=False):
    def _interp(s_from, xi_from, s_to):
        return np.interp(s_to, s_from, xi_from)
    fit   = (s_dat >= smin) & (s_dat <= smax)
    s_fit = s_dat[fit]
    ndof  = fit.sum() * len(ells)
    residuals, weights = [], []
    for ell in ells:
        xi_d = xi_dat[ell][fit]
        xi_m = _interp(s_mock, xi_mock[ell], s_fit)
        residuals.append(xi_d - xi_m)
        if use_jackknife:
            err = err_dat[ell][fit]
            err = np.where((err > 0) & np.isfinite(err), err, 1.0)
            weights.append(err)
        else:
            weights.append(np.ones(fit.sum()))
    delta = np.concatenate(residuals)
    w     = np.concatenate(weights)
    chi2  = float(np.dot(delta / w, delta / w))
    return chi2, ndof


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    log  = setup_logging("compare_clustering_5panel")
    ells = [0, 2]

    data_styles = {
        0: ('o', 'black', r'Data $\xi_0$'),
        2: ('s', 'grey',  r'Data $\xi_2$'),
    }
    mock_ls = {0: '-', 2: '--'}

    colours = cm.tab10(np.linspace(0, 0.9, max(len(args.sig_lgv), 1)))

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes[2, 1].set_visible(False)

    for sample, (row, col) in zip(SAMPLES, LAYOUT):
        ax  = axes[row, col]
        mr  = sample['mr_cut']
        z0  = sample['zmin']
        z1  = sample['zmax']
        is_top_left = (row == 0 and col == 0)

        ymax = {-22.0: 180, -21.5: 120, -21.0: 90, -20.5: 75, -20.0: 70}.get(mr, 120)
        ymin = -ymax * 0.85
        xlim = args.xlim

        title = f"$M_r < {mr}$  |  ${z0} \\leq z \\leq {z1}$"
        ax.set_title(title, fontsize=10)

        # --- Data ---
        df = data_file(mr, z0, z1, args.data_njack)
        data_result = load_result(df, log)
        if data_result is None:
            ax.text(0.5, 0.5, 'Data not found', transform=ax.transAxes,
                    ha='center', va='center', color='red')
            ax.set_xscale('log')
            ax.set_xlim(xlim)
            ax.set_ylim(ymin, ymax)
            continue

        s_dat, xi_dat, err_dat, good_dat, has_cov = extract_multipoles(
            data_result, ells
        )
        s_d = s_dat[good_dat]

        data_handles = []
        for ell in ells:
            fmt, c, lbl = data_styles[ell]
            y    = s_d**2 * xi_dat[ell][good_dat]
            yerr = s_d**2 * err_dat[ell][good_dat]
            if has_cov and np.any(err_dat[ell][good_dat] > 0):
                h = ax.errorbar(s_d, y, yerr=yerr,
                                fmt=fmt, color=c, ms=4, capsize=2, lw=0.8,
                                zorder=10, label=lbl)
            else:
                h, = ax.plot(s_d, y, fmt, color=c, ms=4, zorder=10, label=lbl)
            data_handles.append(h)

        # --- Mocks ---
        mock_handles = []
        for sig_lgv, mock_col in zip(args.sig_lgv, colours):
            mf = mock_file(mr, z0, z1, args.obs_seed, sig_lgv)
            mock_result = load_result(mf, log)
            if mock_result is None:
                continue

            s_m, xi_m, err_m, good_m, _ = extract_multipoles(mock_result, ells)
            chi2, ndof = compute_chi2(
                s_dat, xi_dat, err_dat, s_m, xi_m,
                ells=ells, smin=args.smin, smax=args.smax,
                use_jackknife=args.use_jackknife and has_cov,
            )
            log.info(
                f"Mr{mr} z{z0}-{z1}  sig_lgv={sig_lgv}  "
                f"chi2={chi2:.1f}  ndof={ndof}  chi2/df={chi2/ndof:.2f}"
            )

            s_ = s_m[good_m]
            mock_label = f"sig_lgv={sig_lgv}  $\\chi^2$/df={chi2/ndof:.1f}"
            first = True
            for ell in ells:
                lbl = mock_label if first else f"  $\\xi_{ell}$"
                first = False
                h, = ax.plot(s_, s_**2 * xi_m[ell][good_m],
                             lw=1.5, color=mock_col, ls=mock_ls[ell],
                             label=lbl)
            mock_handles.append(ax.get_lines()[-2])  # solid line handle

        # --- Legends ---
        # Data legend: bottom-left on top-left panel, upper-left elsewhere
        mock_loc = 'upper left'

        if is_top_left:
            data_legend = ax.legend(handles=data_handles, fontsize=7,
                            loc='lower left', framealpha=0.8)
            ax.add_artist(data_legend)
        # Mock legend on every panel
        mock_legend_handles = []
        mock_legend_labels  = []
        for sig_lgv, mock_col in zip(args.sig_lgv, colours):
            mf = mock_file(mr, z0, z1, args.obs_seed, sig_lgv)
            if not Path(mf).exists():
                continue
            # recompute chi2 label — already logged above, just rebuild label
            # Use the lines already on the plot
            pass

        # Simpler: just use ax.legend for mock lines by collecting from plot
        all_lines  = ax.get_lines()
        mock_lines = [l for l in all_lines if l.get_linestyle() in ['-', '--']
                      and l.get_label().startswith('sig_lgv')]
        # Get only solid lines (one per sig_lgv) for the legend entry
        solid_mock = [l for l in mock_lines if l.get_linestyle() == '-']
        ax.legend(handles=solid_mock, fontsize=7, loc=mock_loc, framealpha=0.8)

        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.set_xscale('log')
        ax.set_xlim(xlim)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$s$ [Mpc/$h$]', fontsize=9)
        ax.set_ylabel(r'$s^2\xi_\ell(s)$ [Mpc$^2h^{-2}$]', fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"BGS DR2 data vs Uchuu mocks  (obs_seed={args.obs_seed})",
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    if args.savefig:
        fig.savefig(args.savefig, dpi=150, bbox_inches='tight')
        log.info(f"Figure saved: {args.savefig}")
    else:
        plt.show()

    log.info("compare_clustering_5panel.py complete.")


if __name__ == "__main__":
    main()
